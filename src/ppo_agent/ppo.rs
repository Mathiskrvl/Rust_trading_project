use burn::{
    module::Module,
    optim::{Optimizer, GradientsParams},
    tensor::{
        {Tensor, ElementConversion, Data, Shape,Int, Distribution},
        backend::{Backend, AutodiffBackend},
    },
    record::{PrettyJsonFileRecorder, HalfPrecisionSettings, Recorder},
};
use super::critic::{CriticConfig, Critic};
use super::actor::{ActorConfig, Actor};
fn clip<T>(value: T, min: T, max: T) -> T
where
    T: PartialOrd,
{
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

pub struct PPOAgent<B: Backend> {
    actor: Actor<B>,
    critic: Critic<B>,
    action_range: f64,
}

impl<B: Backend> PPOAgent<B> {
    pub fn init(action_range: f64) -> Self {
        let actor = ActorConfig::new(2048).init();
        let critic = CriticConfig::new(2048).init();
        Self {
            actor,
            critic,
            action_range
        }
    }
    pub fn choose_action(&self, state: Tensor<B, 2>, greedy: bool) -> f64 {
        if greedy {
            let (mean , _) = self.actor.forward(state, self.action_range);
            //clip(mean.flatten::<1>(0, 1).select(0, Tensor::<B, 1, Int>::from_data(Data::zeros([0]))).into_scalar().elem::<f32>(), -self.action_range, self.action_range)
            clip(mean.detach().squeeze::<1>(0).select(0, Tensor::<B, 1, Int>::zeros([1])).into_scalar().elem::<f64>(), -self.action_range, self.action_range)
        }
        else {
            let (mean , std) = self.actor.forward(state, self.action_range);
            let (mean, std) = (mean.detach().squeeze::<1>(0).select(0, Tensor::<B, 1, Int>::zeros([1])).into_scalar().elem::<f64>(), std.detach().squeeze::<1>(0).select(0, Tensor::<B, 1, Int>::zeros([1])).into_scalar().elem::<f64>());//((mean.into_scalar() as f64).elem(), std.into_scalar());
            let mut rng = rand::thread_rng();
            let mut pi = Distribution::Normal(mean, std).sampler(&mut rng);
            clip(pi.sample(),-self.action_range, self.action_range)
        }
    }
    pub fn save_agent(&self, agent_type: &str, filename: &str) {
        std::fs::create_dir(format!("{filename}")).ok();
        self.actor
            .clone()
            .save_file(format!("{agent_type}/{filename}/actor"), &PrettyJsonFileRecorder::<HalfPrecisionSettings>::new())
            .expect("Trained model should be saved successfully");
        self.critic
            .clone()
            .save_file(format!("{agent_type}/{filename}/critic"), &PrettyJsonFileRecorder::<HalfPrecisionSettings>::new())
            .expect("Trained model should be saved successfully");
    }
    pub fn load_agent(agent_type: &String, num_safe: &String, action_range: f64) -> Self {
        let record_actor = PrettyJsonFileRecorder::<HalfPrecisionSettings>::new()
            .load(format!("agent/{agent_type}/trader_{num_safe}/actor").into())
            .expect("Trained model should exist");
        let record_critic = PrettyJsonFileRecorder::<HalfPrecisionSettings>::new()
            .load(format!("agent/{agent_type}/trader_{num_safe}/critic").into())
            .expect("Trained model should exist");
        let actor = ActorConfig::new(2048).init_with::<B>(record_actor);
        let critic = CriticConfig::new(2048).init_with::<B>(record_critic);
        Self {
            actor,
            critic,
            action_range
        }
    }
}

pub struct LearnerPPOAgent<B: AutodiffBackend, OA: Optimizer<Actor<B>, B>, OC: Optimizer<Critic<B>,B>> {
    agent: PPOAgent<B>,
    optim_a: OA,
    optim_c: OC,
    state_buffer: Vec<Tensor<B, 2>>,
    action_buffer: Vec<f64>,
    reward_buffer: Vec<f64>,
    reward_cumulatif_buffer: Vec<f64>,
    gamma: f64,
    epsilon: f64,
    lr_a: f64,
    lr_c: f64,
    actor_update_step: usize,
    critic_update_step: usize,
}

impl<B, OA, OC> LearnerPPOAgent<B, OA, OC> 
where
    B: AutodiffBackend,
    OA: Optimizer<Actor<B>, B>,
    OC: Optimizer<Critic<B>,B>,
{
    pub fn new(agent: PPOAgent<B>, optim_a: OA, optim_c: OC, gamma: f64, epsilon: f64, lr_a: f64, lr_c: f64, actor_update_step: usize, critic_update_step: usize) -> Self {
        //let agent = PPOAgent::init(action_range);
        Self {
            agent,
            optim_a,
            optim_c,
            state_buffer: vec![],
            action_buffer: vec![],
            reward_buffer: vec![],
            reward_cumulatif_buffer: vec![],
            gamma,
            epsilon,
            lr_a,
            lr_c,
            actor_update_step,
            critic_update_step
        }
    }
    pub fn choose_action(&self, state: Tensor<B, 2>, greedy: bool) -> f64 {
        self.agent.choose_action(state, greedy)
    }
    fn actor_train(&mut self, state: Tensor<B,2>, action: Tensor<B, 2>, adv: Tensor<B, 2>, old_pi: (Tensor<B, 2>, Tensor<B,2>)) {
        let (mu, sigma) = self.agent.actor.forward(state, self.agent.action_range);
        let (old_mu, old_sigma) = old_pi;
        // log_prob = -0.5 * (((mean - action) ** 2) / (std ** 2) + torch.log(2 * np.pi * std ** 2))
        let new_log_prob = ((((mu.sub(action.clone())).powf(2f32)).div(sigma.clone().powf(2f32))).add((sigma.powf(2f32).mul_scalar(2f32 * std::f32::consts::PI)).log())).mul_scalar(-0.5);
        let old_log_prob = ((((old_mu.sub(action.clone())).powf(2f32)).div(old_sigma.clone().powf(2f32))).add((old_sigma.powf(2f32).mul_scalar(2f32 * std::f32::consts::PI)).log())).mul_scalar(-0.5);
        let ratio = new_log_prob.sub(old_log_prob).exp();
        let surr = ratio.clone().mul(adv.clone());
        let clamped_adv_ratio = ratio.clamp(1f64 - self.epsilon, 1f64 + self.epsilon).mul(adv);
        let mask = surr.clone().lower_equal(clamped_adv_ratio.clone());
        let min_tensor = surr.mask_where(mask, clamped_adv_ratio);
        let a_loss = min_tensor.mean().neg();
        let grads = a_loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.agent.actor);
        self.agent.actor = self.optim_a.step(self.lr_a, self.agent.actor.clone(), grads);
    }
    fn critic_train(&mut self, cumulatif_r: Tensor<B, 2>, state: Tensor<B, 2>) {
        let advantage = cumulatif_r.sub(self.agent.critic.forward(state));
        let c_loss = advantage.powf(2f32).mean();
        let grads = c_loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.agent.critic);
        self.agent.critic = self.optim_c.step(self.lr_c, self.agent.critic.clone(), grads);
    }
    pub fn update(&mut self) {
        self.calculate_cumulative_reward();
        let s = Tensor::cat(self.state_buffer.clone(), 0);
        let a = Tensor::<B, 1>::from_data(Data::new(self.action_buffer.clone(), Shape::new([self.action_buffer.len(); 1])).convert());
        let r = Tensor::<B, 1>::from_data(Data::new(self.reward_cumulatif_buffer.clone(), Shape::new([self.reward_cumulatif_buffer.len(); 1])).convert());
        let (a, r) = (a.unsqueeze_dim(1), r.unsqueeze_dim(1));
        let (mean, std) = self.agent.actor.forward(s.clone(), self.agent.action_range);
        let (mean, std) = (mean.detach(), std.detach());
        let adv = r.clone().sub(self.agent.critic.forward(s.clone())).detach();
        // update actor
        for _ in 0..self.actor_update_step {
            self.actor_train(s.clone(), a.clone(), adv.clone(), (mean.clone(), std.clone()));
        }
        // update critic
        for _ in 0..self.critic_update_step {
            self.critic_train(r.clone(), s.clone());
        }
        self.state_buffer.clear();
        self.action_buffer.clear();
        self.reward_cumulatif_buffer.clear();
    }
    pub fn store_transition(&mut self, state: Tensor<B, 2>, action: f64, reward: f64) {
        self.action_buffer.push(action);
        self.reward_buffer.push(reward);
        self.state_buffer.push(state);
    }
    fn calculate_cumulative_reward(&mut self) {
        let mut v_s_= 0f64;
        let mut discounted_r = vec![];
        for r in self.reward_buffer.iter().rev() {
            v_s_ = r + self.gamma * v_s_;
            discounted_r.push(v_s_);
        }
        discounted_r.reverse();
        self.reward_cumulatif_buffer.extend(discounted_r);
        self.reward_buffer.clear();
    }
    pub fn record_agent(&self, agent_type: &str, filename: &str) {
        self.agent.save_agent(agent_type, filename);
        let record_a = self.optim_a.to_record();
        let record_c = self.optim_c.to_record();
        std::fs::create_dir(format!("{agent_type}/{filename}/optimizer")).ok();
        let recorder = &PrettyJsonFileRecorder::<HalfPrecisionSettings>::new();
        let _ = recorder.record(record_a, format!("agent/{agent_type}/{filename}/optimizer/optimizer_actor").into());
        let _ = recorder.record(record_c, format!("agent/{agent_type}/{filename}/optimizer/optimizer_critic").into());
    }

}

