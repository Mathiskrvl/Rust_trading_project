#! [warn(clippy::all, clippy::pedantic)]
mod get_data;
//mod test_mnist;
mod encoder;
#[allow(dead_code, unused_imports)]
// use encoder_test::utils::{training, predict};
//use test_mnist::mnist;
use encoder::utils::training;
use get_data::get_data;
use std::thread;
use std::sync::mpsc;

// use burn::tensor::{backend::Backend, Tensor, Shape};
// use burn::backend::{Wgpu, wgpu::AutoGraphicsApi};

// fn example<B: Backend>() {
//     let tensor = Tensor::<B, 2>::ones(Shape::new([3, 3]));
//     let tensor = tensor.unsqueeze_dim::<4>(2);
//     println!("{:?}", tensor.shape());
// }
// type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
fn main() {
    training();
    // example::<MyBackend>();
}


#[allow(dead_code)]
fn thread_miner_bitcoin() {
    let (tx, rx) = mpsc::channel();
    // let data = thread::spawn(move || get_data(tx));
    let miner_btc = thread::spawn(move || {
        get_data(tx, "btcusdt");
    });
    for recu in rx {
        println!("On a reçu {}", recu.time.elapsed().as_millis());
    }
    miner_btc.join().unwrap();
}