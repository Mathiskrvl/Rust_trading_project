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
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.encoder.forward(input);
        let x = self.decoder.forward(x);
        x
    }
}

#[derive(Config, Debug)]
pub struct AutoencoderConfig {
    input_dim_en: usize,
    output_en: usize,
    output_de: usize
}

impl AutoencoderConfig {
    pub fn init<B: Backend>(&self) -> Autoencoder<B> {
        Autoencoder {
            encoder: EncoderConfig::new(self.input_dim_en, self.output_en).init(),
            decoder: DecoderConfig::new(self.output_en, self.output_de).init()
        }
    }
}

