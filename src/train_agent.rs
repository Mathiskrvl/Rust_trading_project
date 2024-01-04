mod ppo_agent;
mod encoder;
mod utils;
mod get_data;
mod run_encoder;
mod reward;
use reward::reward;
use run_encoder::run;
use ppo_agent::ppo::{PPOAgent, LearnerPPOAgent};
use get_data::get_data;
use std::sync::mpsc;
use std::thread;
use burn::{backend::{Wgpu, Autodiff,wgpu::{AutoGraphicsApi, WgpuDevice}}, optim::{AdamConfig, Optimizer}, tensor::Tensor};

use crate::ppo_agent::{actor::Actor, critic::Critic};
fn main() {
    todo!()
}

fn train_agent(gamma: f32, epsilon: f32, lr: f64, actor_update_step: usize, critic_update_step: usize, batch_size: usize, record: usize) {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    std::fs::create_dir("agent").ok();
    std::fs::create_dir("agent/ppo").ok();
    let (tx_data, rx_data) = mpsc::channel();
    let (tx_enc, rx_enc) = mpsc::channel();
    let miner_btc = thread::spawn(move || {
        get_data(tx_data, "btcusdt");
    });
    let encoder = thread::spawn(move || {
        run::<MyBackend>("autoencoder_model_", rx_data, tx_enc);
    });
    let train_agent = thread::spawn(move || {
        // todo add logic for agent.load_agent "agent/ppo/ppo_model_{last_default}.json"
        let agent = PPOAgent::<MyAutodiffBackend>::init(100f32);
        let optim_actor = AdamConfig::new().init::<MyAutodiffBackend, Actor<MyAutodiffBackend>>();
        let optim_critic = AdamConfig::new().init::<MyAutodiffBackend, Critic<MyAutodiffBackend>>();
        let mut learner = LearnerPPOAgent::new(agent, optim_actor, optim_critic, gamma, epsilon, lr, actor_update_step, critic_update_step);
        let (mut compteur_iter, mut compteur_update, mut compteur_record) = (0, 0, 0);
        for recu in rx_enc {
            let state = Tensor::from_inner(recu);
            let action = learner.choose_action(state.clone(), false);
            let reward = reward(); // todo
            compteur_iter += 1;
            learner.store_transition(state, action, reward);
            if compteur_iter >= batch_size {
                compteur_iter = 0;
                learner.update();
                compteur_update += 1;
            }
            if compteur_update >= record {
                compteur_update = 0;
                learner.record_agent("agent/ppo", format!("trader_{compteur_update}").as_str())
            }
        }
    });
    miner_btc.join().unwrap();
    encoder.join().unwrap();
    train_agent.join().unwrap();
}