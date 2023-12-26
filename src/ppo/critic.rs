#! [warn(clippy::all, clippy::pedantic)]
use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, ReLU},
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct Critic<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
    fc4: Linear<B>,
    relu: ReLU
}

impl<B: Backend> Critic<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(input);
        let x = self.relu.forward(x);
        let x = self.fc2.forward(x);
        let x = self.relu.forward(x);
        let x = self.fc3.forward(x);
        let x = self.relu.forward(x);
        let x = self.fc4.forward(x);
        x
    }
}

#[derive(Config, Debug)]
pub struct CriticConfig {
    input_dim: usize,
    #[config(default = 128)]
    hidden_dim: usize,
}

impl CriticConfig {
    pub fn init<B: Backend>(&self) -> Critic<B> {
        Critic {
            fc1: LinearConfig::new(self.input_dim, self.hidden_dim).init(),
            fc2: LinearConfig::new(self.hidden_dim, self.hidden_dim).init(),
            fc3: LinearConfig::new(self.hidden_dim, self.hidden_dim).init(),
            fc4: LinearConfig::new(self.hidden_dim, 1).init(),
            relu: ReLU::new()
        }
    }
    pub fn init_with<B: Backend>(&self, record: CriticRecord<B>) -> Critic<B> {
        Critic {
            fc1: LinearConfig::new(self.input_dim, self.hidden_dim).init_with(record.fc1),
            fc2: LinearConfig::new(self.hidden_dim, self.hidden_dim).init_with(record.fc2),
            fc3: LinearConfig::new(self.hidden_dim, self.hidden_dim).init_with(record.fc3),
            fc4: LinearConfig::new(self.hidden_dim, 1).init_with(record.fc4),
            relu: ReLU::new()
        }
    }
}