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

use rand::Rng;

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
        let actor = ActorConfig::new(128).init();
        let critic = CriticConfig::new(128).init();
        Self {
            actor,
            critic,
            action_range
        }
    }
    pub fn choose_action(&self, state: Tensor<B, 2>, greedy: bool) -> f32 {
        if greedy {
            let (mean , _) = self.actor.forward(state, self.action_range);
            clip(mean.flatten::<1>(0, 1).select(0, Tensor::<B, 1, Int>::from_data(Data::zeros([0]))).into_scalar().elem::<f32>(), -self.action_range, self.action_range)
        }
        else {
            let (mean , std) = self.actor.forward(state, self.action_range);
            let (mean, std) = (mean.flatten::<1>(0, 1).select(0, Tensor::<B, 1, Int>::from_data(Data::zeros([0]))).into_scalar().elem::<f64>(), std.flatten::<1>(0, 1).select(0, Tensor::<B, 1, Int>::from_data(Data::zeros([0]))).into_scalar().elem::<f64>());//((mean.into_scalar() as f64).elem(), std.into_scalar());
            let mut rng = rand::thread_rng();
            let mut pi = Distribution::Normal(mean, std).sampler(&mut rng);
            clip(pi.sample(),-self.action_range, self.action_range)
        }
    }
}

struct LearnerPPOAgent<B: AutodiffBackend, OA: Optimizer<Actor<B>, B>, OC: Optimizer<Critic<B>,B>> {
    agent: PPOAgent<B>,
    optimizer: (OA, OC),
}

impl<B, OA, OC> LearnerPPOAgent<B, OA, OC> 
where
    B: AutodiffBackend,
    OA: Optimizer<Actor<B>, B>,
    OC: Optimizer<Critic<B>,B>,
{
    fn new(optimizer: (OA, OC), action_range: f32) -> Self {
        let agent = PPOAgent::init(action_range);
        Self {
            agent,
            optimizer
        }
    }
    
    fn actor_train() {
        todo!()
    }
    fn critic_train() {
        todo!()
    }
    fn update() {
        todo!()
    }

}