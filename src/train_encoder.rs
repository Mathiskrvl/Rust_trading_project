#! [warn(clippy::all, clippy::pedantic)]
mod get_data;
mod utils;
mod encoder;

use std::{thread, io, sync::mpsc};
use get_data::get_data;
use encoder::autoencoder::AutoencoderConfig;
use utils::{test, data_to_tensor};

use burn::{
    module::Module,
    optim::{AdamConfig, Optimizer, GradientsParams},
    tensor::Tensor,
    backend::{Wgpu, Autodiff,wgpu::{AutoGraphicsApi, WgpuDevice}},
    nn::loss::{MSELoss, Reduction},
    record::{PrettyJsonFileRecorder, HalfPrecisionSettings, Recorder},
};

fn main() {
    println!("Entrez le learning rate :");
    let mut lr_input = String::new();
    io::stdin().read_line(&mut lr_input).expect("Échec de la lecture de l'entrée");
    let lr: f64 = lr_input.trim().parse().expect("Veuillez entrer un nombre valide");
    println!("Entrez init_model (laissez vide si non applicable):");
    let mut init_model_input = String::new();
    io::stdin().read_line(&mut init_model_input).expect("Échec de la lecture de l'entrée");
    let init_model: Option<String> = if init_model_input.trim().is_empty() {
        None
    } else {
        Some(init_model_input.trim().to_string())
    };

    train_encoder(lr, init_model);
    // testing();
}

fn train_encoder(lr: f64, init_model: Option<String>) {
    let (tx, rx) = mpsc::channel();
    let miner_btc = thread::spawn(move || {
        get_data(tx, "btcusdt");
    });
    let train_encoder = thread::spawn(move || {
        type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
        type MyAutodiffBackend = Autodiff<MyBackend>;
        let device = WgpuDevice::default();
        std::fs::create_dir("model").ok();

        let mut model;
        let mut optim;

        if let Some(num_safe) = init_model {
            let record_model = PrettyJsonFileRecorder::<HalfPrecisionSettings>::new()
                .load(format!("model/encoder_record_{num_safe}/autoencoder").into())
                .expect("Trained model should exist");
            model = AutoencoderConfig::new().init_with::<MyAutodiffBackend>(record_model).to_device(&device);
            let record_optimizer =  PrettyJsonFileRecorder::<HalfPrecisionSettings>::new()
                .load(format!("model/encoder_record_{num_safe}/optimizeur").into())
                .expect("Trained model should exist");
            optim = AdamConfig::new().init().load_record(record_optimizer);
            println!("Model load")
        }
        else {
            model = AutoencoderConfig::new().init().to_device(&device);
            optim = AdamConfig::new().init();
            println!("Model create")
        }
        
        let mut state_encoder: Option<(Tensor<MyAutodiffBackend, 2>, Tensor<MyAutodiffBackend, 2>)> = None;
        let mut state_decoder: Option<(Tensor<MyAutodiffBackend, 2>, Tensor<MyAutodiffBackend, 2>)> = None;
        let mut compteur_iter = 0;
        let mut count_save = 0;
        for recu in rx {
            let inputs = data_to_tensor::<MyAutodiffBackend>(recu);
            for input in inputs {
                compteur_iter += 1;
                let (output, new_state_encoder, new_state_decoder) = model.forward(input.clone(), state_encoder, state_decoder);
                state_encoder = Some(new_state_encoder);
                state_decoder = Some(new_state_decoder);
                let loss: Tensor<Autodiff<Wgpu>, 1> = MSELoss::new().forward(output.clone(), input.clone(), Reduction::Auto);
                if compteur_iter % 100 == 0 {
                    println!("Train - Iteration {} - Loss {:.5}", compteur_iter + count_save * 5000, loss.clone().into_scalar());
                }
                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &model);
                model = optim.step(lr, model, grads);
                if compteur_iter % 5000 == 0 {
                    compteur_iter = 0;
                    count_save += 1;
                    let record = optim.to_record();
                    std::fs::create_dir(format!("model/encoder_record_{count_save}")).ok();
                    let recorder = &PrettyJsonFileRecorder::<HalfPrecisionSettings>::new();
                    let _ = recorder.record(record, format!("model/encoder_record_{count_save}/optimizeur").into());
                    model.clone()
                        .save_file(format!("model/encoder_record_{count_save}/autoencoder"), &PrettyJsonFileRecorder::<HalfPrecisionSettings>::new())
                        .expect("Trained model should be saved successfully");
                    model.clone().encoder
                        .save_file(format!("model/encoder_record_{count_save}/encoder"), &PrettyJsonFileRecorder::<HalfPrecisionSettings>::new())
                        .expect("Trained model should be saved successfully");
                    println!("------------------- Model saved -------------------");
                }
            }
        }
        // Création d'un plot pour la courbe d'apprentissage.
    });
    miner_btc.join().unwrap();
    train_encoder.join().unwrap();
}

type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
type MyAutodiffBackend = Autodiff<MyBackend>;
#[allow(dead_code)]
pub fn testing() {
    let device = WgpuDevice::default();
    test::<MyAutodiffBackend>(device);
}