#! [warn(clippy::all, clippy::pedantic)]

use serde_json::Value;
use tungstenite::{connect, Message};
use std::time::{SystemTime, UNIX_EPOCH};
use std::thread;
use std::sync::mpsc::{self, Sender};

fn main() {
    let (tx, rx) = mpsc::channel();
    // let data = thread::spawn(move || get_data(tx));
    let miner = thread::spawn(move || {
        get_data(tx, "btcusdt");
    });
    for recu in rx {
        println!("On a reçu : {}", recu);
    }
    miner.join().unwrap();
}

fn get_data(tx : Sender<String>, symbol: &str) {
    let (mut socket, response) = connect("wss://data-stream.binance.vision/stream").expect("Can't connect");
    println!("{:?}", response);
    let subscribe_message = |symbol: &str, stream_type: &str, id: u8| -> String {
        format!(r#"{{"method": "SUBSCRIBE", "params": ["{}@{}"], "id": {}}}"#, symbol, stream_type, id)
    };
    socket.send(Message::Text(subscribe_message(symbol, "aggTrade", 1))).unwrap();
    socket.send(Message::Text(subscribe_message(symbol, "depth20@100ms", 3))).unwrap();
    socket.send(Message::Text(subscribe_message(symbol, "bookTicker", 4))).unwrap();
    loop {
        let msg = socket.read().expect("Error reading message");
        let msg_str = msg.to_text().expect("Error convert str");
        let json_msg: Value = serde_json::from_str(msg_str).expect("Error JSON");
        if let (Some(data), Some(stream)) = (json_msg.get("data"), json_msg.get("stream").and_then(|v| v.as_str())) {
            println!("{:?}", &json_msg["stream"]);
            if stream.contains("aggTrade") {
                let result = process_agg(data);
                // tx.send(result).unwrap();
            }
            else if stream.contains("bookTicker") {
                let result = process_book(data);
                // tx.send(result).unwrap();
            }
            else if stream.contains("depth20@100ms") {
                let result = process_depth(data);
                break
                // tx.send(result).unwrap();
            }
        }
        else {
            println!("ça n'a pas marché con {:?}", msg_str);
        }
    }
}

fn process_book(data: &Value) {
    println!("book : {:?}", data);
}
fn process_depth(data: &Value) {
    //let (asks, bids) = (&data["asks"], &data["bids"]);
    // for ask in asks {

    // }
    // println!("depth : {:?}", ask);
    println!("depth : {:?}", data);
}

fn process_agg(data: &Value) {
    // if let (Some(symbol), Some(quantity), Some(price), Some(time)) = (
    //     data.get("s"),
    //     data.get("q").and_then(|v| v.as_str()),
    //     data.get("p").and_then(|v| v.as_str()),
    //     data.get("T").and_then(|v| v.as_u64())) {
    //         counter +=1;
    //         if counter >= 100 {
    //             let temps = SystemTime::now().duration_since(UNIX_EPOCH).expect("Systeme time error").as_millis() as u64;
    //             let diff = temps - time;
    //             //println!("A {time}: {quantity} de {} a été acheté à {price}, retard {} milli_sec", symbol.to_string(), diff);
    //             tx.send(format!("A {time}: {quantity} de {} a été acheté à {price}, retard {diff} milli_sec, ", symbol.to_string())).unwrap();
    //             counter = 0;
    //         }
    println!("depth : {:?}", data);
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
