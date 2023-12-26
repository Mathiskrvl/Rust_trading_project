use burn::optim::{Adam, Optimizer};
use burn::{tensor::backend::{Backend, AutodiffBackend},
        config::Config,
        optim::AdamConfig};

use super::critic::{CriticConfig, Critic};
use super::actor::{ActorConfig, Actor};


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

struct PPOAgent<B: Backend, OA: Optimizer<Actor<B>, B>> {
    actor: Actor<B>,
    critic: Critic<B>,
    optim_a: OA,
    optim_c: Adam<B>,
}

impl<B: Backend, OA: Optimizer<Actor<B>, B>> PPOAgent<B, OA> {

}