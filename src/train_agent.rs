mod ppo_agent;
mod encoder;
mod utils;
mod get_data;
mod run_encoder;

use utils::PorteFeuille;
use run_encoder::run_encoder;
use ppo_agent::ppo::{PPOAgent, LearnerPPOAgent};
use get_data::get_data;
use std::{thread, io, sync::mpsc};

use burn::{
        backend::{Wgpu, Autodiff,wgpu::AutoGraphicsApi}, 
        optim::{AdamConfig, Optimizer},
        tensor::Tensor,
        record::{PrettyJsonFileRecorder, HalfPrecisionSettings, Recorder}};
use crate::ppo_agent::{actor::Actor, critic::Critic};

fn main() {
    // println!("Entrez le type du d'agent souhaitez (disponible seulement PPO) : ");
    // let mut init_agent_type = String::new();
    // io::stdin().read_line(&mut init_agent_type).expect("Échec de la lecture de l'entrée");
    // let agent_type: String = init_agent_type.trim().parse().expect("Veuillez entrer un nombre valide");
    let agent_type: String = String::from("ppo");

    println!("Entrez init_agent (laissez vide si non applicable): ");
    let mut init_agent_input = String::new();
    io::stdin().read_line(&mut init_agent_input).expect("Échec de la lecture de l'entrée");
    let init_agent: Option<String> = if init_agent_input.trim().is_empty() {
        None
    } else {
        Some(init_agent_input.trim().to_string())
    };

    println!("Entrez le numero du model d'encoder : ");
    let mut init_encoder_input = String::new();
    io::stdin().read_line(&mut init_encoder_input).expect("Échec de la lecture de l'entrée");
    let init_encoder: String = init_encoder_input.trim().parse().expect("Veuillez entrer un nombre valide");

    // update de l'agent toute les 2048 changements d'états soit toute les 200 secondes. Cela laisse 2048 possibilité d'action.
    // record de l'agent toute les 18 updates soit ~1h
    // On estime qu'il faut environ 100 000 update soit 232 jours pour avoir un model fiable, dout l'importance de rajouter 
    // à ce programme des système de curiosité et multi-agents pour converger + vite vers un bot de trading fiable.
    train_agent(agent_type, init_agent, init_encoder, 100f64, 0.1, 0.9, 0.2, 1e-4, 2e-4, 10, 10, 2048, 18)
}

fn train_agent(agent_type: String, init_agent: Option<String>, init_encoder: String, action_range: f64, phi: f64, gamma: f64, epsilon: f64, lr_a: f64, lr_c: f64, actor_update_step: usize, critic_update_step: usize, batch_size: usize, record: usize) {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    std::fs::create_dir("agent").ok();
    std::fs::create_dir(format!("agent/{agent_type}")).ok();
    let (tx_data, rx_data) = mpsc::channel();
    let (tx_enc, rx_enc) = mpsc::channel();
    let miner_btc = thread::spawn(move || {
        get_data(tx_data, "btcusdt");
    });
    let encoder = thread::spawn(move || {
        run_encoder::<MyBackend>(init_encoder, rx_data, tx_enc);
    });
    let train_agent = thread::spawn(move || {
        // configurer la quantity de usdt et btc de départ. Un peu près 2000$ diviser entre 50% usdt et 50% btc
        let mut wallet = PorteFeuille::new(2000f64);
        let agent;
        let optim_actor;
        let optim_critic;
        if let Some(num_safe) = init_agent {
            agent = PPOAgent::<MyAutodiffBackend>::load_agent(&agent_type, &num_safe, action_range);
            let record_optimizer_actor =  PrettyJsonFileRecorder::<HalfPrecisionSettings>::new()
                .load(format!("agent/{agent_type}/trader_{num_safe}/optimizeur/optimizer_actor").into())
                .expect("Trained model should exist");
            let record_optimizer_critic =  PrettyJsonFileRecorder::<HalfPrecisionSettings>::new()
                .load(format!("agent/{agent_type}/trader_{num_safe}/optimizeur/optimizer_critic").into())
                .expect("Trained model should exist");
            optim_actor = AdamConfig::new().init().load_record(record_optimizer_actor);
            optim_critic = AdamConfig::new().init().load_record(record_optimizer_critic);
            println!("-------------- Agent load --------------");
        }
        else {
            agent = PPOAgent::<MyAutodiffBackend>::init(action_range);
            optim_actor = AdamConfig::new().init::<MyAutodiffBackend, Actor<MyAutodiffBackend>>();
            optim_critic = AdamConfig::new().init::<MyAutodiffBackend, Critic<MyAutodiffBackend>>();
            println!("-------------- Agent create --------------");
        }
        let mut learner = LearnerPPOAgent::new(agent, optim_actor, optim_critic, gamma, epsilon, lr_a, lr_c, actor_update_step, critic_update_step);
        let (mut compteur_iter, mut compteur_update, mut compteur_record) = (0, 0, 0);
        for recu in rx_enc {
            let state = Tensor::from_inner(recu);
            let action = learner.choose_action(state.clone(), false);
            wallet.trade(action, phi);
            let reward = wallet.reward();
            compteur_iter += 1;
            learner.store_transition(state, action, reward);
            if compteur_iter >= batch_size || reward > 10f64 {
                compteur_iter = 0;
                learner.update();
                compteur_update += 1;
            }
            if compteur_update >= record || reward > 10f64{
                compteur_update = 0;
                compteur_record += 1;
                learner.record_agent(agent_type.as_str(), format!("trader_{compteur_record}").as_str());
                println!("------------------- Agent saved -------------------");
            }
        }
    });
    miner_btc.join().unwrap();
    encoder.join().unwrap();
    train_agent.join().unwrap();
}