use burn::backend::wgpu::tensor;
use burn::module::AutodiffModule;
use burn::optim::{Adam, Optimizer};
use burn::tensor::ops::TensorOps;
use burn::tensor::{Tensor, ElementConversion, Data};
use burn::tensor::Int;
use burn::tensor::Distribution;
use burn::{tensor::backend::{Backend, AutodiffBackend},
        config::Config,
        optim::AdamConfig};
use burn::optim::GradientsParams;

use super::critic::{CriticConfig, Critic, self};
use super::actor::{ActorConfig, Actor, self};


// #[derive(Config)]
// struct PPOAgentConfig {
//     actor: ActorConfig,
//     critic: CriticConfig,
//     optimizer_a: AdamConfig,
//     optimizer_b: AdamConfig,
//     #[config(default = 1e-4)]
//     lr_a: f64,
//     #[config(default = 2e-4)]
//     lr_c: f64,
// }
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

struct PPOAgent<B: Backend> {
    actor: Actor<B>,
    critic: Critic<B>,
    action_range: f32,
}

impl<B: Backend> PPOAgent<B> {
    pub fn init(action_range: f32) -> Self {
        let actor = ActorConfig::new(2048).init();
        let critic = CriticConfig::new(2048).init();
        Self {
            actor,
            critic,
            action_range
        }
    }
    pub fn choose_action(&self, state: Tensor<B, 2>, greedy: bool) -> f32 {
        if greedy {
            let (mean , _) = self.actor.forward(state, self.action_range);
            //clip(mean.flatten::<1>(0, 1).select(0, Tensor::<B, 1, Int>::from_data(Data::zeros([0]))).into_scalar().elem::<f32>(), -self.action_range, self.action_range)
            clip(mean.detach().squeeze::<1>(0).select(0, Tensor::<B, 1, Int>::from_data(Data::zeros([0]))).into_scalar().elem::<f32>(), -self.action_range, self.action_range)
        }
        else {
            let (mean , std) = self.actor.forward(state, self.action_range);
            let (mean, std) = (mean.detach().squeeze::<1>(0).select(0, Tensor::<B, 1, Int>::from_data(Data::zeros([0]))).into_scalar().elem::<f64>(), std.detach().squeeze::<1>(0).select(0, Tensor::<B, 1, Int>::from_data(Data::zeros([0]))).into_scalar().elem::<f64>());//((mean.into_scalar() as f64).elem(), std.into_scalar());
            let mut rng = rand::thread_rng();
            let mut pi = Distribution::Normal(mean, std).sampler(&mut rng);
            clip(pi.sample(),-self.action_range, self.action_range)
        }
    }
    pub fn save_agent() {
        todo!()
    }
    pub fn load_agent() {
        todo!()
    }
}

struct LearnerPPOAgent<B: AutodiffBackend, OA: Optimizer<Actor<B>, B>, OC: Optimizer<Critic<B>,B>> {
    agent: PPOAgent<B>,
    optim_a: OA,
    optim_c: OC,
    state_buffer: Vec<Tensor<B, 1>>,
    action_buffer: Vec<f32>,
    reward_buffer: Vec<f32>,
    reward_cumulatif_buffer: Vec<f32>,
    gamma: f32,
    epsilon: f32,
    lr: f64,
}

impl<B, OA, OC> LearnerPPOAgent<B, OA, OC> 
where
    B: AutodiffBackend,
    OA: Optimizer<Actor<B>, B>,
    OC: Optimizer<Critic<B>,B>,
{
    pub fn new(agent: PPOAgent<B>, optim_a: OA, optim_c: OC, gamma: f32, epsilon: f32, lr: f64) -> Self {
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
            lr,
        }
    }
    fn actor_train(&mut self, state: Tensor<B,2>, action: Tensor<B, 2>, adv: Tensor<B, 2>, old_pi: (Tensor<B, 2>, Tensor<B,2>)) {
        let (mu, sigma) = self.agent.actor.forward(state, self.agent.action_range);
        let (old_mu, old_sigma) = old_pi;
        // log_prob = -0.5 * (((mean - action) ** 2) / (std ** 2) + torch.log(2 * np.pi * std ** 2))
        let new_log_prob = ((((mu.sub(action.clone())).powf(2f32)).div(sigma.clone().powf(2f32))).add((sigma.powf(2f32).mul_scalar(2f32 * std::f32::consts::PI)).log())).mul_scalar(-0.5);
        let old_log_prob = ((((old_mu.sub(action.clone())).powf(2f32)).div(old_sigma.clone().powf(2f32))).add((old_sigma.powf(2f32).mul_scalar(2f32 * std::f32::consts::PI)).log())).mul_scalar(-0.5);
        let ratio = new_log_prob.sub(old_log_prob).exp();
        let surr = ratio.clone().mul(adv.clone());
        let clamped_adv_ratio = ratio.clamp(1f32 - self.epsilon, 1f32 + self.epsilon).mul(adv);
        let mask = surr.clone().lower_equal(clamped_adv_ratio.clone());
        let min_tensor = surr.mask_where(mask, clamped_adv_ratio);
        let a_loss = min_tensor.mean().neg();
        let grads = a_loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.agent.critic);
        self.agent.actor = self.optim_a.step(self.lr, self.agent.actor.clone(), grads);
    }
    fn critic_train(&mut self, cumulatif_r: Tensor<B, 1>, state: Tensor<B, 2>) {
        let advantage = cumulatif_r.sub(self.agent.critic.forward(state).squeeze(0));
        let c_loss = advantage.powf(2f32).mean();
        let grads = c_loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.agent.critic);
        self.agent.critic = self.optim_c.step(self.lr, self.agent.critic.clone(), grads);
    }
    pub fn update(&self) {
        todo!()
    }
    pub fn store_transition(&mut self, state: Tensor<B, 2>, action: f32, reward: f32) {
        self.action_buffer.push(action);
        self.reward_buffer.push(reward);
        self.state_buffer.push(state.squeeze::<1>(0));
    }
    fn calculate_cumulative_reward(&mut self, next_state: Tensor<B, 2>, done: bool) {
        let mut v_s_: f32;
        if done {
            v_s_ = 0f32;
        }
        else {
            let critics = self.agent.critic.forward(next_state).squeeze::<1>(0);
            v_s_ = critics.detach().select(0, Tensor::<B, 1, Int>::from_data(Data::zeros([0]))).into_scalar().elem::<f32>()
        }
        let mut discounted_r = vec![];
        for r in self.reward_buffer.iter().rev() {
            v_s_ = r + self.gamma * v_s_;
            discounted_r.push(v_s_);
        }
        discounted_r.reverse();
        self.reward_cumulatif_buffer.extend(discounted_r);
        self.reward_buffer.clear();
    }

}