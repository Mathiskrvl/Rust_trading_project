use std::time::Instant;
use crate::get_data::MyData;
use crate::encoder::autoencoder::AutoencoderConfig;
use burn::{
    module::Module,
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

pub fn reward() -> f32{
    todo!()
}

pub fn trade() {
    todo!()
}


#[allow(dead_code)]
pub fn test_encoder<B: AutodiffBackend>(device: B::Device) {
    let mut model= AutoencoderConfig::new().init().to_device(&device);
    let mut optim = AdamConfig::new().init();
    let mut state_encoder: Option<(Tensor<B, 2>, Tensor<B, 2>)> = None;
    let mut state_decoder: Option<(Tensor<B, 2>, Tensor<B, 2>)> = None;
    let time = Instant::now();
    for epoch in 1..6 {
        for iteration in 0..1000 {
            let inputs = Tensor::<B, 4>::random( [1, 1, 40, 2], Distribution::Uniform(0., 50.));
            let (output, new_state_encoder, new_state_decoder) = model.forward(inputs.clone(), state_encoder, state_decoder);
            state_encoder = Some(new_state_encoder);
            state_decoder = Some(new_state_decoder);
            let loss = MSELoss::new().forward(output.clone(), inputs.clone(), Reduction::Auto);
            println!("[Train - Epoch {} iteration {} - Loss {:.5}", epoch, iteration, loss.clone().into_scalar());
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(1e-4, model, grads);
        }
    };
    println!("{}", time.elapsed().as_secs());
}

