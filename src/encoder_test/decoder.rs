#! [warn(clippy::all, clippy::pedantic)]
use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig,},
    tensor::{ backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    fc1: Linear<B>,
}

impl<B: Backend> Decoder<B> {
    pub fn forward(&self, output_encoder: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(output_encoder);
        x
    }
}

#[derive(Config, Debug)]
pub struct DecoderConfig {
    output_encoder: usize,
    fc1_dim: usize
}

impl DecoderConfig {
    pub fn init<B: Backend>(&self) -> Decoder<B> {
        Decoder {
            fc1: LinearConfig::new(self.output_encoder, self.fc1_dim).init(),
        }
    }
    pub fn init_with<B: Backend>(&self, record: DecoderRecord<B>) -> Decoder<B> {
        Decoder {
            fc1: LinearConfig::new(self.output_encoder, self.fc1_dim).init_with(record.fc1),
        }
    }
}
