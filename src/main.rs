#! [warn(clippy::all, clippy::pedantic)]

use serde_json::Value;
use tungstenite::{connect, Message};
use std::time::{SystemTime, UNIX_EPOCH};
//use std::thread;

fn main() {
    let (mut socket, response) = connect("wss://data-stream.binance.vision/stream").expect("Can't connect");
    println!("{:?}", response);
    let subscribe_message = |symbol: &str, stream_type: &str, id: u8| -> String {
        format!(r#"{{"method": "SUBSCRIBE", "params": ["{}@{}"], "id": {}}}"#, symbol, stream_type, id)
    };
    socket.send(Message::Text(subscribe_message("btcusdt", "aggTrade", 1))).unwrap();
    socket.send(Message::Text(subscribe_message("ethusdt", "aggTrade", 2))).unwrap();
    socket.send(Message::Text(subscribe_message("btcusdt", "depth@100ms", 3))).unwrap();
    socket.send(Message::Text(subscribe_message("btcusdt", "bookTicker", 4))).unwrap();
    let mut counter = 0;
    loop {
        // let debut2 = Instant::now();
        let msg = socket.read().expect("Error reading message");
        if msg.is_ping() {
            println!("I receive a ping");
            //socket.send(Message::Pong(vec![3])).expect("Le pong a beug bg");
        }
        // let duree2 = debut2.elapsed();
        // println!("socket time: {}", duree2.as_micros());
        // let debut = Instant::now();
        let msg_str = msg.to_text().expect("Error convert str");
        let json_msg: Value = serde_json::from_str(msg_str).expect("Error JSON");
        if let Some(data) = json_msg.get("data") {
            // if let (Some(symbol), Some(sub)) = (data.get("s"), data.get("e")) {
            //     println!("{}, {}", symbol.to_string(), sub.to_string());
            // }
            if let (Some(symbol), Some(quantity), Some(price), Some(time)) = (
                data.get("s"),
                data.get("q").and_then(|v| v.as_str()),
                data.get("p").and_then(|v| v.as_str()),
                data.get("T").and_then(|v| v.as_u64())) {
                    counter +=1;
                    if counter >= 100 {
                        let temps = SystemTime::now().duration_since(UNIX_EPOCH).expect("Systeme time error").as_millis() as u64;
                        let diff = temps - time;
                        println!("A {time}: {quantity} de {} a été acheté à {price}, nous sommes à {}", symbol.to_string(), diff);
                        counter = 0;
                    }
                // println!("A {time}: {quantity} de {} a été acheté à {price}", symbol.to_string());
                // let duree = debut.elapsed();
                // println!("process time :{}", duree.as_micros());
            }
        }
        else {
            println!("ça n'a pas marché con {:?}", msg_str);
        }
        // if counter == 10 {
        //     break;
        // }
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
