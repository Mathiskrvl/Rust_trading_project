#! [warn(clippy::all, clippy::pedantic)]
use burn::{
    config::Config,
    module::Module,
    tensor::{backend::Backend ,Tensor}
};
use super::decoder::{Decoder, DecoderConfig};
use super::encoder::{Encoder, EncoderConfig};

#[derive(Module, Debug)]
pub struct Autoencoder<B: Backend> {
    pub encoder: Encoder<B>,
    decoder: Decoder<B>
}

impl<B: Backend> Autoencoder<B> {
    pub fn forward(&self, input: Tensor<B, 4>, state_encoder: Option<(Tensor<B, 2>, Tensor<B, 2>)>, state_decoder: Option<(Tensor<B, 2>, Tensor<B, 2>)>) -> (Tensor<B, 4>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let (x, ce, he) = self.encoder.forward(input , state_encoder);
        let (x, cd, hd) = self.decoder.forward(x, state_decoder);
        (x, ce, he, cd, hd)
    }
}

#[derive(Config, Debug)]
pub struct AutoencoderConfig {}

impl AutoencoderConfig {
    pub fn init<B: Backend>(&self) -> Autoencoder<B> {
        Autoencoder {
            encoder: EncoderConfig::new().init(),
            decoder: DecoderConfig::new().init()
        }
    }
    // pub fn init_with<B: Backend>(&self, record: AutoencoderRecord<B>) -> Autoencoder<B> {
    //     Autoencoder {
    //         encoder: EncoderConfig::new().init_with(record.encoder),
    //         decoder: DecoderConfig::new().init_with(record.decoder)
    //     }
    // }
}

