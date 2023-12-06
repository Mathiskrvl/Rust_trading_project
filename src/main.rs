#! [warn(clippy::all, clippy::pedantic)]
mod get_data;
// mod test_mnist;
mod encoder;

use encoder::train::training;
//use test_mnist::mnist;
use get_data::get_data;
use std::thread;
use std::sync::mpsc;

fn main() {
    training();
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