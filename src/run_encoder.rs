use crate::get_data::MyData;
use crate::utils::data_to_tensor;
use crate::encoder::encoder::EncoderConfig;
use burn::{
        tensor::{backend::Backend, Tensor},
        record::{Recorder, PrettyJsonFileRecorder, HalfPrecisionSettings}};

use std::sync::mpsc::{Receiver, Sender};

pub fn run_encoder<B: Backend>(encoder_file: String, rx:  Receiver<MyData>, tx: Sender<Tensor<B, 2>>) {
    // if std::fs::metadata(format!("model/encoder/{}", encoder_file)).is_ok()     
    let record = PrettyJsonFileRecorder::<HalfPrecisionSettings>::new()
        .load(format!("model/encoder_record_{encoder_file}/encoder").into())
        .expect("Trained model should exist");
    let model= EncoderConfig::new().init_with::<B>(record);
    let mut state: Option<(Tensor<B, 2>, Tensor<B, 2>)> = None;
    for recu in rx {
        let input_encoder = data_to_tensor::<B>(recu);
        if let Some(input) = input_encoder {
            let (output, c, h) = model.forward(input, state);
            state = Some((c, h));
            tx.send(output).unwrap();
        }
    }
}