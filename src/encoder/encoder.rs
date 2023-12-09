use burn::{
    config::Config,
    module::Module,
    nn::{
        lstm::Lstm,
        conv::{Conv2d, Conv2dConfig},
        Linear, LinearConfig, ReLU, Dropout, DropoutConfig, LstmConfig},
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    lstm: Lstm<B>,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: ReLU,
    dropout: Dropout,
}

impl<B: Backend> Encoder<B> {
    pub fn forward(&self, order_book: Tensor<B, 4>, state: Option<(Tensor<B, 2>, Tensor<B, 2>)>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let x = self.conv1.forward(order_book);
        let x = self.dropout.forward(x);
        let x = self.conv2.forward(x);
        let x = self.activation.forward(x);

        let x = x.flatten::<2>(1, 3);
        let x = x.unsqueeze_dim(1);

        let (c, h) = self.lstm.forward(x, state);
        let (c, h) = (c.squeeze(1), h.squeeze(1));

        let x = self.linear1.forward(h.clone());
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        let x = self.linear2.forward(x);
        (x, c, h)
    }
}

#[derive(Config, Debug)]
pub struct EncoderConfig {
    #[config(default= 0.2)]
    dropout: f64
}

impl EncoderConfig {
    pub fn init<B: Backend>(&self) -> Encoder<B> {
        Encoder {
            conv1: Conv2dConfig::new([2, 16], [1, 2]).init(),
            conv2: Conv2dConfig::new([16, 32], [20, 1]).init(),
            lstm: LstmConfig::new(32, 64, true).init(),
            linear1: LinearConfig::new(64, 128).init(),
            linear2: LinearConfig::new(128, 256).init(),
            activation: ReLU::new(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}