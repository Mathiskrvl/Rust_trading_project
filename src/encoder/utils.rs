#! [warn(clippy::all, clippy::pedantic)]
use burn::{
    module::{AutodiffModule, Module},
    config::Config,
    tensor::{
        backend::{AutodiffBackend, Backend},
        Tensor, Distribution},
    optim::{AdamConfig, Optimizer, GradientsParams}, 
    nn::loss::{MSELoss, Reduction},
    backend::{Autodiff, Wgpu, wgpu::{AutoGraphicsApi, WgpuDevice}},
    record::{CompactRecorder, Recorder}
};

use super::autoencoder::AutoencoderConfig;


#[derive(Config)]
pub struct AutoencoderTrainingConfig {
    #[config(default = 30)]
    pub num_epochs: usize,
    #[config(default = 40)]
    pub seed: u64,
    #[config(default = 1e-4)]
    pub lr: f64,
    pub model: AutoencoderConfig,
    pub optimizer: AdamConfig,
}

pub fn run<B: AutodiffBackend>(artifact_dir: &str, device: B::Device) {
    // Create the configuration.
    let config_model = AutoencoderConfig::new(8, 128, 8);
    let config_optimizer = AdamConfig::new();
    let config = AutoencoderTrainingConfig::new(config_model, config_optimizer);

    std::fs::create_dir(artifact_dir).ok();
    config.save(format!("{artifact_dir}/config.json")).expect("Config should be saved successfully");

    B::seed(config.seed);

    let mut model= config.model.init().to_device(&device);
    let mut optim = config.optimizer.init();
    let mut multiplicator_lr = 1.;
    for epoch in 1..config.num_epochs + 1 {
        if epoch%5 == 0 {
            multiplicator_lr = multiplicator_lr * 1e-1;
        }
        for iteration in 0..3000 {
            let train_tensor = Tensor::<B, 2>::random( [64,8], Distribution::Uniform(0., 1e3));
            let output = model.forward(train_tensor.clone()).to_device(&device);
            let loss = MSELoss::new().forward(output.clone(), train_tensor.clone(), Reduction::Auto);

            println!("[Train - Epoch {} iteration {} - Loss {:.3}", epoch, iteration, loss.clone().into_scalar());

            let grads = loss.backward();
            // Gradients linked to each parameter of the model.
            let grads = GradientsParams::from_grads(grads, &model);
            // Update the model using the optimizer.
            model = optim.step(config.lr * multiplicator_lr, model, grads);
        }

        let model_valid = model.valid();
        let test_tensor = Tensor::<B::InnerBackend, 2>::random( [1,8], Distribution::Uniform(0., 1e3));
        let output = model_valid.forward(test_tensor.clone());
        let loss = MSELoss::new().forward(output.clone(), test_tensor.clone(), Reduction::Auto);

        println!(
            "[Valid - Epoch {} - Loss {}", epoch, loss.clone().into_scalar());

    model.clone().save_file(format!("{artifact_dir}/autoencoder"), &CompactRecorder::new()).expect("Trained model should be saved successfully");
    model.encoder.clone().save_file(format!("{artifact_dir}/encoder"), &CompactRecorder::new()).expect("Trained model should be saved successfully");
    };
}

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device) {
    let config = AutoencoderTrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/autoencoder").into())
        .expect("Trained model should exist");

    let model = config.model.init_with::<B>(record).to_device(&device);
    let test_tensor = Tensor::<B, 2>::random( [1,8], Distribution::Uniform(0., 1e3));
    println!("{test_tensor}");
    let output = model.forward(test_tensor.clone());

    println!("Predicted {} Expected {}", output, test_tensor);
}

type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

pub fn training() {
    let device = WgpuDevice::default();
    run::<MyAutodiffBackend>("./model/encoder", device);
}

pub fn predict() {
    let device = WgpuDevice::default();
    infer::<MyBackend>("./model/encoder", device);
}