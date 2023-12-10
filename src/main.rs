#! [warn(clippy::all, clippy::pedantic)]
mod get_data;
//mod test_mnist;
mod utils;
mod encoder;

use std::thread;
use std::sync::mpsc;
use get_data::get_data;
use encoder::autoencoder::AutoencoderConfig;
use utils::grads;

use burn::{
    module::Module,
    config::Config,
    optim::{AdamConfig, Optimizer},
    tensor::{Tensor, backend::Backend},
    backend::{Wgpu, Autodiff,wgpu::{AutoGraphicsApi, WgpuDevice}},
};

fn main() {
    thread_miner_bitcoin();
}

#[derive(Config)]
pub struct AutoencoderTrainingConfig {
    #[config(default = 30)]
    pub seed: u64,
    #[config(default = 1e-4)]
    pub lr: f64,
    pub model: AutoencoderConfig,
    pub optimizer: AdamConfig,
}

#[allow(dead_code)]
fn thread_miner_bitcoin() {
    let (tx, rx) = mpsc::channel();
    // let data = thread::spawn(move || get_data(tx));
    let miner_btc = thread::spawn(move || {
        get_data(tx, "btcusdt");
    });
    let train_encoder = thread::spawn(move || {
        type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
        type MyAutodiffBackend = Autodiff<MyBackend>;
        let device = WgpuDevice::default();
        let config_model = AutoencoderConfig::new();
        let config_optimizer = AdamConfig::new();
        let config = AutoencoderTrainingConfig::new(config_model, config_optimizer);
        std::fs::create_dir("model").ok();
        std::fs::create_dir("model/autoencoder_model").ok();
        std::fs::create_dir("model/encoder_model").ok();
        config.save("model/autoencoder_model/config.json").expect("Config should be saved successfully");
        MyAutodiffBackend::seed(config.seed);
        let mut model= config.model.init().to_device(&device);
        let mut optim = config.optimizer.init();
        let mut state_encoder: Option<(Tensor<MyAutodiffBackend, 2>, Tensor<MyAutodiffBackend, 2>)> = None;
        let mut state_decoder: Option<(Tensor<MyAutodiffBackend, 2>, Tensor<MyAutodiffBackend, 2>)> = None;
        for recu in rx {
            let (grads,new_state_encoder, new_state_decoder)  = grads::<MyAutodiffBackend>(recu, model.clone(), state_encoder, state_decoder);
            state_encoder = new_state_encoder;
            state_decoder = new_state_decoder;
            model = optim.step(config.lr, model, grads);
        }
    });
    miner_btc.join().unwrap();
    train_encoder.join().unwrap();
}