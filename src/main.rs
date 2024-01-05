mod ppo_agent;
mod encoder;
mod utils;
mod get_data;
mod run_encoder;

use utils::PorteFeuille;
use run_encoder::run_encoder;
use ppo_agent::ppo::PPOAgent;
use get_data::get_data;
use std::{thread, io, sync::mpsc};
use burn::{
    backend::{Wgpu, wgpu::AutoGraphicsApi},
    tensor::Tensor,
};
fn main() {
    let agent_type: String = String::from("ppo");

    println!("Entrez le numero de l'agent : ");
    let mut init_agent_input = String::new();
    io::stdin().read_line(&mut init_agent_input).expect("Échec de la lecture de l'entrée");
    let init_agent: String = init_agent_input.trim().parse().expect("Veuillez entrer un nombre valide");

    println!("Entrez le numero du model d'encoder : ");
    let mut init_encoder_input = String::new();
    io::stdin().read_line(&mut init_encoder_input).expect("Échec de la lecture de l'entrée");
    let init_encoder: String = init_encoder_input.trim().parse().expect("Veuillez entrer un nombre valide");

    run_agent(agent_type, init_agent, init_encoder, 100f64, 0.1f64);
}

fn run_agent(agent_type: String, init_agent: String, init_encoder: String, action_range: f64, phi: f64) {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    let (tx_data, rx_data) = mpsc::channel();
    let (tx_enc, rx_enc) = mpsc::channel();
    let miner_btc = thread::spawn(move || {
        get_data(tx_data, "btcusdt");
    });
    let encoder = thread::spawn(move || {
        run_encoder::<MyBackend>(init_encoder, rx_data, tx_enc);
    });
    let run_agent = thread::spawn(move || {
        // configurer la quantity de usdt et btc de départ. Un peu près 2000$ diviser entre 50% usdt et 50% btc
        let mut wallet = PorteFeuille::new(2000f64);
        let agent = PPOAgent::<MyBackend>::load_agent(&agent_type, &init_agent, action_range);
        println!("-------------- Agent load --------------");
        for state in rx_enc {
            let action = agent.choose_action(state, true);
            wallet.trade(action, phi);
            println!("pnl : {}", wallet.reward());
        }

    });
    miner_btc.join().unwrap();
    encoder.join().unwrap();
    run_agent.join().unwrap();
}