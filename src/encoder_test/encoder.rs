#! [warn(clippy::all, clippy::pedantic)]
use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, ReLU},
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    relu: ReLU
}

impl<B: Backend> Encoder<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(input);
        let x = self.relu.forward(x);
        let x = self.fc2.forward(x);
        x
    }
}

#[derive(Config, Debug)]
pub struct EncoderConfig {
    input_dim: usize,
    #[config(default = 64)]
    fc1_dim: usize,
    fc2_dim: usize
}

impl EncoderConfig {
    pub fn init<B: Backend>(&self) -> Encoder<B> {
        Encoder {
            fc1: LinearConfig::new(self.input_dim, self.fc1_dim).init(),
            fc2: LinearConfig::new(self.fc1_dim, self.fc2_dim).init(),
            relu: ReLU::new()
        }
    }
    pub fn init_with<B: Backend>(&self, record: EncoderRecord<B>) -> Encoder<B> {
        Encoder {
            fc1: LinearConfig::new(self.input_dim, self.fc1_dim).init_with(record.fc1),
            fc2: LinearConfig::new(self.fc1_dim, self.fc2_dim).init_with(record.fc2),
            relu: ReLU::new()
        }
    }
}