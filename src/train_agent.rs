mod ppo;
mod encoder;
mod utils;
mod get_data;
mod run_encoder;
use run_encoder::run;
use ppo::ppo::{PPOAgent, LearnerPPOAgent};
use get_data::get_data;
use std::sync::mpsc;
use std::thread;
use burn::{backend::{Wgpu, Autodiff,wgpu::{AutoGraphicsApi, WgpuDevice}}, optim::{AdamConfig, Optimizer}};

use crate::ppo::{actor::Actor, critic::Critic};
pub fn main() {
    todo!()
}
fn train_agent(gamma: f32, epsilon: f32, lr: f64, actor_update_step: usize, critic_update_step: usize,) {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    // let device = WgpuDevice::default();
    let (tx_data, rx_data) = mpsc::channel();
    let (tx_enc, rx_enc) = mpsc::channel();
    let miner_btc = thread::spawn(move || {
        get_data(tx_data, "btcusdt");
    });
    let encoder = thread::spawn(move || {
        run::<MyBackend>("autoencoder_model_", rx_data, tx_enc);
    });
    let train_agent = thread::spawn(move || {
        let agent = PPOAgent::<MyAutodiffBackend>::init(100f32);
        let optim_actor = AdamConfig::new().init::<MyAutodiffBackend, Actor<MyAutodiffBackend>>();
        let optim_critic = AdamConfig::new().init::<MyAutodiffBackend, Critic<MyAutodiffBackend>>();
        let learner = LearnerPPOAgent::new(agent, optim_actor, optim_critic, gamma, epsilon, lr, actor_update_step, critic_update_step);
        for recu in rx_enc {
            todo!()
        }
    });
    miner_btc.join().unwrap();
    encoder.join().unwrap();
    train_agent.join().unwrap();
}