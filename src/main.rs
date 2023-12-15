#! [warn(clippy::all, clippy::pedantic)]
mod get_data;
//mod test_mnist;
mod utils;
mod encoder;

use std::thread;
use std::sync::mpsc;
use get_data::get_data;
use encoder::autoencoder::AutoencoderConfig;
use utils::{run, data_to_tensor};

use burn::{
    module::Module,
    config::Config,
    optim::{AdamConfig, Optimizer, GradientsParams},
    tensor::{Tensor, backend::Backend},
    backend::{Wgpu, Autodiff,wgpu::{AutoGraphicsApi, WgpuDevice}},
    nn::loss::{MSELoss, Reduction},
    record::{PrettyJsonFileRecorder, HalfPrecisionSettings},
};

fn main() {
    thread_miner_bitcoin();
    // training();
}

#[derive(Config)]
pub struct AutoencoderTrainingConfig {
    #[config(default = 42)]
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
    let test_encoder = thread::spawn(move || {
        type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
        type MyAutodiffBackend = Autodiff<MyBackend>;
        let device = WgpuDevice::default();
        let config_model = AutoencoderConfig::new();
        let config_optimizer = AdamConfig::new();
        let config = AutoencoderTrainingConfig::new(config_model, config_optimizer);
        std::fs::create_dir("model").ok();
        std::fs::create_dir("model/autoencoder").ok();
        std::fs::create_dir("model/encoder").ok();
        config.save("model/autoencoder/config.json").expect("Config should be saved successfully");
        MyAutodiffBackend::seed(config.seed);
        println!("model create");
        let mut model= config.model.init().to_device(&device);
        let mut optim = config.optimizer.init();
        let mut state_encoder: Option<(Tensor<MyAutodiffBackend, 2>, Tensor<MyAutodiffBackend, 2>)> = None;
        let mut state_decoder: Option<(Tensor<MyAutodiffBackend, 2>, Tensor<MyAutodiffBackend, 2>)> = None;
        let mut compteur_iter = 0;
        let mut count_save = 0;
        for recu in rx {
            // let test = Tensor::<MyAutodiffBackend, 4>::random( [1, 2, 20, 2], Distribution::Uniform(0., 5e1));
            let inputs = data_to_tensor::<MyAutodiffBackend>(recu);
            for input in inputs {
                compteur_iter += 1;
                let (output, new_state_encoder, new_state_decoder) = model.forward(input.clone(), state_encoder, state_decoder);
                state_encoder = Some(new_state_encoder);
                state_decoder = Some(new_state_decoder);
                let loss = MSELoss::new().forward(output.clone(), input.clone(), Reduction::Auto);
                if compteur_iter % 100 == 0 {
                    println!("Train - Iteration {} - Loss {:.5}", compteur_iter, loss.clone().into_scalar());
                }
                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &model);
                model = optim.step(config.lr, model, grads);
                if compteur_iter % 50000 == 0 {
                    count_save +=1;
                    let model_cloned = model.clone();
                    thread::spawn(move || {
                        model_cloned.clone()
                            .save_file(format!("model/autoencoder/autoencoder_model_{count_save}"), &PrettyJsonFileRecorder::<HalfPrecisionSettings>::new())
                            .expect("Trained model should be saved successfully");
                        model_cloned.encoder
                            .save_file(format!("model/encoder/encoder_model_{count_save}"), &PrettyJsonFileRecorder::<HalfPrecisionSettings>::new())
                            .expect("Trained model should be saved successfully");
                        println!("------------------- Model saved -------------------");
                    });
                    // model.clone().save_file(format!("model/autoencoder/autoencoder_model_{}", count_save), &CompactRecorder::new()).expect("Trained model should be saved successfully");
                    // model.encoder.clone().save_file(format!("model/encoder_model_{}", count_save), &CompactRecorder::new()).expect("Trained model should be saved successfully");
                    // println!("Model saved");
                }
            }
        }
    });
    miner_btc.join().unwrap();
    test_encoder.join().unwrap();
}

type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

pub fn training() {
    let device = WgpuDevice::default();
    run::<MyAutodiffBackend>("./model", device);
}