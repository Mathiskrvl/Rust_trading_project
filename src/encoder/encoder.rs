use burn::{
    config::Config,
    module::Module,
    nn::{
        lstm::Lstm,
        conv::{Conv2d, Conv2dConfig},
        Linear, LinearConfig, ReLU, Dropout, DropoutConfig, LstmConfig, LayerNorm, LayerNormConfig},
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    conv1: Conv2d<B>,
    lstm: Lstm<B>,
    prelinear: Linear<B>,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: ReLU,
    dropout: Dropout,
    norm: LayerNorm<B>,
}

impl<B: Backend> Encoder<B> {
    pub fn forward(&self, order_book: Tensor<B, 4>, state: Option<(Tensor<B, 2>, Tensor<B, 2>)>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        // shape = (1, 1, 40, 2)
        let x = self.conv1.forward(order_book);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        // shape = (1, 16, 40, 1)
        let x = x.flatten::<2>(1, 3);
        // shape = (1, 640)
        let x = self.prelinear.forward(x);
        let x = self.activation.forward(x);
        // shape = (1, 1024)
        let x = self.norm.forward(x);
        // shape = (1, 1024)
        let x = x.unsqueeze_dim(1);
        // shape = (1, 1, 1024)
        let (c, h) = self.lstm.forward(x, state);
        let (c, h) = (c.squeeze(1).detach(), h.squeeze(1).detach());
        // shape = (1, 1024)
        let x = self.linear1.forward(h.clone());
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        // shape = (1, 2048)
        let x = self.linear2.forward(x);
        // shape = (1, 2048)
        (x, c, h)
    }
}

#[derive(Config, Debug)]
pub struct EncoderConfig {
    #[config(default= 0.1)]
    dropout: f64
}

impl EncoderConfig {
    pub fn init<B: Backend>(&self) -> Encoder<B> {
        Encoder {
            conv1: Conv2dConfig::new([1, 16], [1, 2]).init(),
            lstm: LstmConfig::new(1024, 1024, true).init(),
            prelinear: LinearConfig::new(640, 1024).init(),
            linear1: LinearConfig::new(1024, 2048).init(),
            linear2: LinearConfig::new(2048, 2048).init(),
            activation: ReLU::new(),
            dropout: DropoutConfig::new(self.dropout).init(),
            norm: LayerNormConfig::new(1024).init(),
        }
    }
    pub fn init_with<B: Backend>(&self, record: EncoderRecord<B>) -> Encoder<B> {
        Encoder {
            conv1: Conv2dConfig::new([1, 16], [1, 2]).init_with(record.conv1),
            lstm: LstmConfig::new(1024, 1024, true).init_with(record.lstm),
            prelinear: LinearConfig::new(640, 1024).init_with(record.prelinear),
            linear1: LinearConfig::new(1024, 2048).init_with(record.linear1),
            linear2: LinearConfig::new(2048, 2048).init_with(record.linear2),
            activation: ReLU::new(),
            dropout: DropoutConfig::new(self.dropout).init(),
            norm: LayerNormConfig::new(1024).init_with(record.norm),
        }
    }
}