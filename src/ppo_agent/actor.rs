#! [warn(clippy::all, clippy::pedantic)]
use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, ReLU},
    tensor::{backend::Backend, Tensor, activation::tanh},
};

#[derive(Module, Debug)]
pub struct Actor<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
    fc4: Linear<B>,
    mean: Linear<B>,
    log_std: Linear<B>,
    relu: ReLU,
}

impl<B: Backend> Actor<B> {
    pub fn forward(&self, input: Tensor<B, 2>, action_range: f64) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let x = self.fc1.forward(input);
        let x = self.relu.forward(x);
        let x = self.fc2.forward(x);
        let x = self.relu.forward(x);
        let x = self.fc3.forward(x);
        let x = self.relu.forward(x);
        let x = self.fc4.forward(x);

        let mean = tanh(self.mean.forward(x.clone())).mul_scalar(action_range);

        let log_std = self.log_std.forward(x);
        // let log_std = clamp(log_std, min_std, max_std);
        let std = log_std.exp();

        (mean, std)
    }
}

#[derive(Config, Debug)]
pub struct ActorConfig {
    input_dim: usize,
    #[config(default = 1)]
    num_action: usize,
    #[config(default = 128)]
    hidden_dim: usize,
}

impl ActorConfig {
    pub fn init<B: Backend>(&self) -> Actor<B> {
        Actor {
            fc1: LinearConfig::new(self.input_dim, self.hidden_dim).init(),
            fc2: LinearConfig::new(self.hidden_dim, self.hidden_dim).init(),
            fc3: LinearConfig::new(self.hidden_dim, self.hidden_dim).init(),
            fc4: LinearConfig::new(self.hidden_dim, self.hidden_dim).init(),
            mean: LinearConfig::new(self.hidden_dim, self.num_action).init(),
            log_std: LinearConfig::new(self.hidden_dim, self.num_action).init(),
            relu: ReLU::new(),
        }
    }
    pub fn init_with<B: Backend>(&self, record: ActorRecord<B>) -> Actor<B> {
        Actor {
            fc1: LinearConfig::new(self.input_dim, self.hidden_dim).init_with(record.fc1),
            fc2: LinearConfig::new(self.hidden_dim, self.hidden_dim).init_with(record.fc2),
            fc3: LinearConfig::new(self.hidden_dim, self.hidden_dim).init_with(record.fc3),
            fc4: LinearConfig::new(self.hidden_dim, self.hidden_dim).init_with(record.fc4),
            mean: LinearConfig::new(self.hidden_dim, self.num_action).init_with(record.mean),
            log_std: LinearConfig::new(self.hidden_dim, self.num_action).init_with(record.log_std),
            relu: ReLU::new(),
        }
    }
}