use crate::get_data::MyData;
use crate::utils::data_to_tensor;
use crate::encoder::encoder::EncoderConfig;
use burn::{
        tensor::{backend::Backend, Tensor},
        record::{Recorder, PrettyJsonFileRecorder, HalfPrecisionSettings}};

use std::sync::mpsc::{Receiver, Sender};

pub fn run<B: Backend>(encoder_file: &str, rx:  Receiver<MyData>, tx: Sender<Tensor<B, 2>>) {
    // if std::fs::metadata(format!("model/encoder/{}", encoder_file)).is_ok()     
    let record = PrettyJsonFileRecorder::<HalfPrecisionSettings>::new()
        .load(format!("model/encoder/{encoder_file}").into())
        .expect("Trained model should exist");
    let model= EncoderConfig::new().init_with::<B>(record);
    let mut state: Option<(Tensor<B, 2>, Tensor<B, 2>)> = None;
    for recu in rx {
        let inputs = data_to_tensor::<B>(recu);
        for input in inputs {
            let (output, c, h) = model.forward(input.clone(), state);
            state = Some((c, h));
            tx.send(output).unwrap();
        }
    }
}