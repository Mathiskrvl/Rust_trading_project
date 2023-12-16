#! [warn(clippy::all, clippy::pedantic)]

use serde_json::Value;
use tungstenite::{connect, Message};
use std::time::{SystemTime, UNIX_EPOCH, Instant, Duration};
use std::sync::mpsc::Sender;

pub fn get_data(tx : Sender<MyData>, symbol: &str) {
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
    let mut my_data = MyData::new();
    loop {
        // trouver une solution pour ne pas attendre un message indéfiniment lorsque nous n'avons pu de wiki
        let msg = socket.read().expect("Error reading message");
        let msg_str = msg.to_text().expect("Error convert str");
        let json_msg: Value = serde_json::from_str(msg_str).expect("Error JSON");
        if let (Some(data), Some(stream)) = (json_msg.get("data"), json_msg.get("stream").and_then(|v| v.as_str())) {
            if stream.contains("trade") {
                my_data.process_agg(data, true);
            }
            else if stream.contains("depth20@100ms") {
                my_data.process_depth(data);
            }
        }
        else if msg.is_ping() {
            println!("on a le ping le reuf {:?}", msg);
        }
        else {
            println!("ça n'a pas marché con {:?}", msg_str);
        }
        if Instant::now().duration_since(my_data.time) >= Duration::from_millis(100) {
            tx.send(my_data).unwrap();
            my_data = MyData::new();
        }
    }
}

#[derive(Debug)]
pub struct MyData {
    pub time: Instant,
    pub trade: Vec<(f32, f32, bool)>,
    pub orderbook: Vec<[[f32; 2]; 40]>
}

impl MyData {
    fn new() -> Self {
        Self { time: Instant::now(), trade: vec![], orderbook: vec![] }
    }
    fn process_depth(&mut self, data: &Value) {
        if let (Some(asks), Some(bids)) = (data.get("asks").and_then(|v| v.as_array()), data.get("bids").and_then(|v| v.as_array())) {
            let max_bid = bids[0][0].as_str().unwrap().parse::<f32>().unwrap();
            let min_ask= asks[0][0].as_str().unwrap().parse::<f32>().unwrap();
            let mut tableau2d = [[0f32; 2]; 40];
            for (i, bid) in bids.iter().rev().enumerate() {
                let (bided, quantity) = (bid[0].as_str().unwrap().parse::<f32>().unwrap() - (min_ask + max_bid) / 2f32, bid[1].as_str().unwrap().parse::<f32>().unwrap());
                tableau2d[i] = [bided, quantity];
            }
            for (i, ask) in asks.iter().enumerate() {
                let (asked, quantity) = (ask[0].as_str().unwrap().parse::<f32>().unwrap() - (min_ask + max_bid) / 2f32, ask[1].as_str().unwrap().parse::<f32>().unwrap());
                tableau2d[i + 20] = [asked, - quantity];
            }
            self.orderbook.push(tableau2d);
        }
    }
    fn process_agg(&mut self ,data: &Value, retard: bool) {
        if let (Some(maker), Some(quantity), Some(price), Some(time)) = (
            data.get("m").and_then(|v| v.as_bool()),
            data.get("q").and_then(|v| v.as_str()),
            data.get("p").and_then(|v| v.as_str()),
            data.get("T").and_then(|v| v.as_u64())) {
                let quantity = quantity.parse::<f32>().unwrap();
                let price = price.parse::<f32>().unwrap();
                self.trade.push((price, quantity, maker));
                if retard {
                    let temps = SystemTime::now().duration_since(UNIX_EPOCH).expect("Systeme time error").as_millis() as u64;
                    let retard = temps - time;
                    if retard >= 500 {
                        println!("retard : {}", retard);
                    }
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
