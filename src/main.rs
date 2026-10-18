mod engine;
mod nn;

use engine::Value;
use nn::{Gradients, MLP};

fn main() {
    let inputs: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets: Vec<f32> = vec![-1.0, 1.0, 1.0, -1.0];

    let mlp = MLP::new(2, vec![4, 1]);
    let params = mlp.parameters();
    let lr = 0.02;
    let steps = 3000;

    for step in 0..steps {
        let mut loss = Value::new(0.0);
        let mut correct = 0;

        for (row, &target) in inputs.iter().zip(targets.iter()) {
            let x: Vec<Value> = row.iter().map(|&v| Value::new(v)).collect();
            let pred = mlp.forward(&x)[0].clone();

            if (pred.data() > 0.0) == (target > 0.0) {
                correct += 1;
            }

            let diff = pred + Value::new(-target);
            loss = loss + diff.pow(2.0);
        }
        mlp.zero_grad();
        loss.backward();
        for p in &params {
            p.set_data(p.data() - lr * p.grad());
        }

        if step % 50 == 0 || step == steps - 1 {
            println!("step {step:4}  loss {:.4}  correct {correct}/4", loss.data());
        }
    }

    println!();
    for (row, &target) in inputs.iter().zip(targets.iter()) {
        let x: Vec<Value> = row.iter().map(|&v| Value::new(v)).collect();
        let pred = mlp.forward(&x)[0].data();
        println!("input {row:?}  target {target:+.0}  prediction {pred:+.3}");
    }
}