use burn::{
    config::Config,
    module::Module,
    nn::{
        lstm::Lstm,
        conv::{ConvTranspose2d, ConvTranspose2dConfig},
        Linear, LinearConfig, ReLU, Dropout, DropoutConfig, LstmConfig},
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    convt1: ConvTranspose2d<B>,
    convt2: ConvTranspose2d<B>,
    lstm: Lstm<B>,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: ReLU,
    dropout: Dropout,
}

impl<B: Backend> Decoder<B> {
    pub fn forward(&self, encoder_output: Tensor<B, 2>, state: Option<(Tensor<B, 2>, Tensor<B, 2>)>) -> (Tensor<B, 4>, Tensor<B, 2>, Tensor<B, 2>) {
        let x = self.linear2.forward(encoder_output);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        let x = self.linear1.forward(x);

        let x = x.unsqueeze_dim(1);

        let (c, h) = self.lstm.forward(x, state);
        let (c, h) = (c.squeeze::<2>(1), h.squeeze::<2>(1));
        let x = h.clone().unsqueeze_dim::<4>(2);

        let x = self.convt2.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        let x = self.convt1.forward(x);
        (x, c, h)
    }
}

#[derive(Config, Debug)]
pub struct DecoderConfig {
    #[config(default= 0.2)]
    dropout: f64
}

impl DecoderConfig {
    pub fn init<B: Backend>(&self) -> Decoder<B> {
        Decoder {
            linear2: LinearConfig::new(256, 128).init(),
            linear1: LinearConfig::new(128, 64).init(),
            lstm: LstmConfig::new( 64, 32, true).init(),
            convt2: ConvTranspose2dConfig::new([32, 16], [20, 1]).init(),
            convt1: ConvTranspose2dConfig::new([16, 2], [1, 2]).init(),
            activation: ReLU::new(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}