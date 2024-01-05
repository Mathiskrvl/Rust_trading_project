use std::time::Instant;
use reqwest::{self, blocking::Client};
use serde_json::Value;
use crate::get_data::MyData;
use crate::encoder::autoencoder::AutoencoderConfig;
use burn::{
    module::Module,
    tensor::{
        backend::{AutodiffBackend, Backend},
        Tensor, Data, Distribution},
    optim::{GradientsParams, AdamConfig, Optimizer}, 
    nn::loss::{MSELoss, Reduction},
};

pub fn data_to_tensor<B: Backend>(data: MyData) -> Tensor<B, 4> {
    let orderbook = Tensor::<B, 2>::from_data(Data::<f32, 2>::from(data.orderbook.last().unwrap().clone()).convert()).unsqueeze::<4>();
    orderbook
}

pub fn data_to_vec_tensor<B: Backend>(data: MyData) -> Vec<Tensor<B, 4>> {
    let orderbooks = data.orderbook
        .iter()
        .map(|orderbook: &[[f32; 2]; 40]| Data::<f32, 2>::from(orderbook.clone()))
        .map(|data| Tensor::<B, 2>::from_data(data.convert()))
        .map(|tensor| tensor.unsqueeze::<4>())
        .collect();
    orderbooks
}

pub struct PorteFeuille {
    client : Client,
    btc: f64,
    usdt: f64,
    first_state: f64,
}

impl PorteFeuille {
    pub fn new(usdt: f64) -> Self {
        let url = String::from("https://data-api.binance.vision/api/v3/ticker/price?symbol=BTCUSDT");
        let client = Client::new();
        let response = client.get(&url)
                             .send()
                             .expect("erreur lors de la requete");
        let res_json: Value = response.json().expect("erreur lors de la conversion en JSON");
        let price = res_json
            .get("price")
            .expect("Iln'y a pas de price dans cette réponse")
            .as_str()
            .expect("Erreur de conversion")
            .parse::<f64>()
            .expect("Nombre pas valide");
        let moitie = usdt / 2f64;
        Self {
            client,
            btc : moitie / price,
            usdt: moitie,
            first_state : usdt,
        }
    }
    fn state(&self) -> f64 {
        let price = self.get_btc_price();
        self.btc * price + self.usdt
    }
    pub fn trade(&mut self, quantity: f64, phi: f64) {
        if (quantity >= 0f64 + phi) || (quantity <= 0f64 - phi)  {
            let price = self.get_btc_price();
            if quantity > 0f64 {
                let quantity_usdt = self.usdt * (quantity / 100f64);
                self.btc +=  quantity_usdt / price;
                self.usdt -= quantity_usdt;
                println!("We buy {quantity_usdt} of BTC");
            }
            else if quantity < 0f64 {
                let quantity_btc = self.btc * (quantity / 100f64);
                self.usdt += quantity_btc * price;
                self.btc -=  quantity_btc;
                println!("We sell {} of BTC", quantity_btc * quantity);
            }
        }
        else {
            println!("Waiting good oportunities");
        }
    }
    fn get_btc_price(&self) -> f64 {
        let url = String::from("https://data-api.binance.vision/api/v3/ticker/price?symbol=BTCUSDT");
        let response = self.client.get(&url)
                             .send()
                             .expect("erreur lors de la requete");
        let res_json: Value = response.json().expect("erreur lors de la conversion en JSON");
        res_json
            .get("price")
            .expect("Iln'y a pas de price dans cette réponse")
            .as_str()
            .expect("Erreur de conversion")
            .parse::<f64>()
            .expect("Nombre pas valide")
    }
    pub fn reward(&self) -> f64 {
        let state = self.state();
        (state - self.first_state) / self.first_state
    }
}



#[allow(dead_code)]
pub fn test_encoder<B: AutodiffBackend>(device: B::Device) {
    let mut model= AutoencoderConfig::new().init().to_device(&device);
    let mut optim = AdamConfig::new().init();
    let mut state_encoder: Option<(Tensor<B, 2>, Tensor<B, 2>)> = None;
    let mut state_decoder: Option<(Tensor<B, 2>, Tensor<B, 2>)> = None;
    let time = Instant::now();
    for epoch in 1..6 {
        for iteration in 0..1000 {
            let inputs = Tensor::<B, 4>::random( [1, 1, 40, 2], Distribution::Uniform(0., 50.));
            let (output, new_state_encoder, new_state_decoder) = model.forward(inputs.clone(), state_encoder, state_decoder);
            state_encoder = Some(new_state_encoder);
            state_decoder = Some(new_state_decoder);
            let loss = MSELoss::new().forward(output.clone(), inputs.clone(), Reduction::Auto);
            println!("[Train - Epoch {} iteration {} - Loss {:.5}", epoch, iteration, loss.clone().into_scalar());
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(1e-4, model, grads);
        }
    };
    println!("{}", time.elapsed().as_secs());
}

