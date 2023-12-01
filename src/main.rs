#! [warn(clippy::all, clippy::pedantic)]

use serde_json::Value;
use tungstenite::{connect, Message};
use std::time::{SystemTime, UNIX_EPOCH, Instant, Duration};
use std::thread;
use std::sync::mpsc::{self, Sender};

fn main() {
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

fn get_data(tx : Sender<Data>, symbol: &str) {
    let (mut socket, response) = connect("wss://data-stream.binance.vision/stream").expect("Can't connect");
    println!("{:?}", response);
    let subscribe_message = |symbol: &str, stream_type: &str, id: u8| -> String {
        format!(r#"{{"method": "SUBSCRIBE", "params": ["{}@{}"], "id": {}}}"#, symbol, stream_type, id)
    };
    socket.send(Message::Text(subscribe_message(symbol, "trade", 1))).unwrap();
    socket.send(Message::Text(subscribe_message(symbol, "depth20@100ms", 2))).unwrap();
    // socket.send(Message::Text(subscribe_message(symbol, "bookTicker", 4))).unwrap();
    // socket.read().expect("Subscribe_error");
    socket.read().expect("Subscribe_error");
    socket.read().expect("Subscribe_error");
    let mut my_data = Data::new();
    loop {
        let msg = socket.read().expect("Error reading message");
        let msg_str = msg.to_text().expect("Error convert str");
        let json_msg: Value = serde_json::from_str(msg_str).expect("Error JSON");
        if let (Some(data), Some(stream)) = (json_msg.get("data"), json_msg.get("stream").and_then(|v| v.as_str())) {
            // println!("{:?}", &json_msg["stream"]);
            if stream.contains("trade") {
                process_agg(&mut my_data, data, true);
            }
            else if stream.contains("depth20@100ms") {
                process_depth(&mut my_data, data);
            }
        }
        else {
            println!("ça n'a pas marché con {:?}", msg_str);
        }
        if Instant::now().duration_since(my_data.time) >= Duration::from_millis(500) {
            tx.send(my_data).unwrap();
            my_data = Data::new();
        }
    }
}

#[derive(Debug)]
struct Data {
    time: Instant,
    trade: Vec<(f32, f32, bool)>,
    orderbook: Vec<(Vec<(f32, f32)>, Vec<(f32, f32)>)>
}

impl Data {
    fn new() -> Self {
        Self { time: Instant::now(), trade: vec![], orderbook: vec![] }
    }
}

fn process_depth(my_data:&mut Data, data: &Value) {
    if let (Some(asks), Some(bids)) = (data.get("asks").and_then(|v| v.as_array()), data.get("bids").and_then(|v| v.as_array())) {
        let mut data_bids: Vec<(f32, f32)> = vec![];
        for bid in bids.iter().rev() {
            let (bided, quantity) = (bid[0].as_str().unwrap().parse::<f32>().unwrap(), bid[1].as_str().unwrap().parse::<f32>().unwrap());
            data_bids.push((bided, quantity));
        }
        let mut data_asks: Vec<(f32, f32)> = vec![];
        for ask in asks.iter() {
            let (asked, quantity) = (ask[0].as_str().unwrap().parse::<f32>().unwrap(), ask[1].as_str().unwrap().parse::<f32>().unwrap());
            data_asks.push((asked, quantity));
        }
        // for (price, quantity) in data_bids.iter() {
        //     println!("bids : {price} : {quantity}");
        // }
        // for (price, quantity) in data_asks.iter() {
        //     println!("asks : {price} : {quantity}");
        // }
        my_data.orderbook.push((data_bids, data_asks));
    }
}

fn process_agg(my_data:&mut Data ,data: &Value, retard: bool) {
    if let (Some(maker), Some(quantity), Some(price), Some(time)) = (
        data.get("m").and_then(|v| v.as_bool()),
        data.get("q").and_then(|v| v.as_str()),
        data.get("p").and_then(|v| v.as_str()),
        data.get("T").and_then(|v| v.as_u64())) {
            let quantity = quantity.parse::<f32>().unwrap();
            let price = price.parse::<f32>().unwrap();
            my_data.trade.push((price, quantity, maker));
            if retard {
                let temps = SystemTime::now().duration_since(UNIX_EPOCH).expect("Systeme time error").as_millis() as u64;
                let retard = temps - time;
                if retard >= 500 {
                    println!("retard : {}", retard);
                }
            }
        }
}


// use reqwest;
// use serde_json::Value;
// use tokio;
// #[tokio::main]
// async fn requete() {
//     let mut url = String::from("https://data-api.binance.vision");
//     url.push_str("/api/v3/trades?symbol=BTCUSDT&limit=1");
//     // let mut url = String::from("https://testnet.binancefuture.com");
//     // url.push_str("/fapi/v1/time");
//     let client = reqwest::Client::new();
//     let response = client.get(&url).send().await.expect("erreur lors de la requete");
//     let res_json = response.json::<Value>().await.expect("erreur lors de la requete");
//     println!("{:?}", &res_json);
// }
