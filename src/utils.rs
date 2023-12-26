use std::time::Instant;

// mod encoder;
// mod get_data;
use crate::get_data::MyData;
use crate::encoder::autoencoder::AutoencoderConfig;
use burn::{
    module::Module,
    config::Config,
    tensor::{
        backend::{AutodiffBackend, Backend},
        Tensor, Data, Distribution},
    optim::{GradientsParams, AdamConfig, Optimizer}, 
    nn::loss::{MSELoss, Reduction},
};

pub fn data_to_tensor<B: Backend>(data: MyData) -> Vec<Tensor<B, 4>> {
    let orderbooks = data.orderbook
        .iter()
        .map(|orderbook: &[[f32; 2]; 40]| Data::<f32, 2>::from(orderbook.clone()))
        .map(|data| Tensor::<B, 2>::from_data(data.convert()))
        .map(|tensor| tensor.unsqueeze::<4>())
        .collect();
    //let orderbooks = Tensor::cat(orderbooks, 0);
    orderbooks
}

#[derive(Config)]
pub struct AutoencoderTrainingConfig {
    #[config(default = 5)]
    pub num_epochs: usize,
    #[config(default = 40)]
    pub seed: u64,
    #[config(default = 1e-4)]
    pub lr: f64,
    pub model: AutoencoderConfig,
    pub optimizer: AdamConfig,
}

pub fn run<B: AutodiffBackend>(device: B::Device) {
    // Create the configuration.
    let config_model = AutoencoderConfig::new();
    let config_optimizer = AdamConfig::new();
    let config = AutoencoderTrainingConfig::new(config_model, config_optimizer);

    // std::fs::create_dir(artifact_dir).ok();
    // std::fs::create_dir(format!("{artifact_dir}/autoencoder_model")).ok();
    // std::fs::create_dir(format!("{artifact_dir}/encoder_model")).ok();
    // config.save(format!("{artifact_dir}/autoencoder_model/config.json")).expect("Config should be saved successfully");

    B::seed(config.seed);

    let mut model= config.model.init().to_device(&device);
    let mut optim = config.optimizer.init();
    let mut state_encoder: Option<(Tensor<B, 2>, Tensor<B, 2>)> = None;
    let mut state_decoder: Option<(Tensor<B, 2>, Tensor<B, 2>)> = None;

    let time = Instant::now();
    for epoch in 1..config.num_epochs + 1 {
        for iteration in 0..1000 {
            let inputs = Tensor::<B, 4>::random( [1, 1, 40, 2], Distribution::Uniform(0., 50.));
            let (output, new_state_encoder, new_state_decoder) = model.forward(inputs.clone(), state_encoder, state_decoder);
            state_encoder = Some(new_state_encoder);
            state_decoder = Some(new_state_decoder);
            let loss = MSELoss::new().forward(output.clone(), inputs.clone(), Reduction::Auto);
            println!("[Train - Epoch {} iteration {} - Loss {:.3}", epoch, iteration, loss.clone().into_scalar());
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(config.lr, model, grads);
        }
        // let model_valid = model.valid();
        // let inputs = Tensor::<B::InnerBackend, 4>::ones([1,2,20,2]);
        // let output = model_valid.forward(inputs.clone(), state_encoder.clone(), state_decoder.clone());
        // let loss = MSELoss::new().forward(output.clone(), inputs.clone(), Reduction::Auto);

        // println!(
        //     "[Valid - Epoch {} - Loss {}", epoch, loss.clone().into_scalar());
    };
    println!("{}", time.elapsed().as_secs());
    // model.clone().save_file(format!("{artifact_dir}/autoencoder_model"), &CompactRecorder::new()).expect("Trained model should be saved successfully");
    // model.encoder.clone().save_file(format!("{artifact_dir}/encoder_model"), &CompactRecorder::new()).expect("Trained model should be saved successfully");
}

