mod engine;
use engine::{Value};

fn main() {
    println!("Hello, world!");
    5.0 as f32 * Value::new(3.0);
}
