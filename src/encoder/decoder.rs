use burn::{
    config::Config,
    module::Module,
    nn::{
        lstm::Lstm,
        conv::{ConvTranspose2d, ConvTranspose2dConfig},
        Linear, LinearConfig, ReLU, Dropout, DropoutConfig, LstmConfig, LayerNorm, LayerNormConfig},
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    convt1: ConvTranspose2d<B>,
    lstm: Lstm<B>,
    prelinear: Linear<B>,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: ReLU,
    dropout: Dropout,
    norm: LayerNorm<B>,
}

impl<B: Backend> Decoder<B> {
    pub fn forward(&self, encoder_output: Tensor<B, 2>, state: Option<(Tensor<B, 2>, Tensor<B, 2>)>) -> (Tensor<B, 4>, Tensor<B, 2>, Tensor<B, 2>) {
        // shape = (1, 2048)
        let x = self.linear2.forward(encoder_output);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        // shape = (1, 2048)
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        // shape = (1, 1024)
        let x = self.norm.forward(x);
        // shape = (1, 1024)
        let x = x.unsqueeze_dim(1);
        // shape = (1, 1, 1024)
        let (c, h) = self.lstm.forward(x, state);
        let (c, h) = (c.squeeze::<2>(1).detach(), h.squeeze::<2>(1).detach());
        // shape = (1, 1024)
        let x = self.prelinear.forward(h.clone());
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        // shape = (1, 640)
        let x = x.reshape([1, 16, 40, 1]);
        // shape = (1, 16, 40, 1)
        let x = self.convt1.forward(x);
        // shape = (1, 1, 40, 2)
        (x, c, h)
    }
}

#[derive(Config, Debug)]
pub struct DecoderConfig {
    #[config(default= 0.1)]
    dropout: f64
}

impl DecoderConfig {
    pub fn init<B: Backend>(&self) -> Decoder<B> {
        Decoder {
            prelinear: LinearConfig::new(1024, 640).init(),
            linear2: LinearConfig::new(2048, 2048).init(),
            linear1: LinearConfig::new(2048, 1024).init(),
            lstm: LstmConfig::new( 1024, 1024, true).init(),
            convt1: ConvTranspose2dConfig::new([16, 1], [1, 2]).init(),
            activation: ReLU::new(),
            dropout: DropoutConfig::new(self.dropout).init(),
            norm: LayerNormConfig::new(1024).init(),
        }
    }
}