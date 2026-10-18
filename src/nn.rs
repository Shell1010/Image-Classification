
use crate::Value;
use rand::distr::{Distribution, Uniform};


pub trait Gradients {
    fn parameters(&self) -> Vec<Value>;

    fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }
}

#[derive(Clone)]
pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
    non_linear: bool
}

impl Neuron {
    pub fn new(inputs: usize, non_linear: bool) -> Self {
        let sample = Uniform::new(-1.0, 1.0).unwrap();
        let mut rng = rand::rng();
        let weights: Vec<Value> = (0..inputs)
            .map(|_| Value::new(sample.sample(&mut rng)))
            .collect();
    
        Neuron { weights, bias: Value::new(0.0), non_linear }
    }

    pub fn forward(&self, val: &[Value]) -> Value {
        let sum = self.weights.iter().zip(val)
            .fold(self.bias.clone(), |acc, (w, v)| acc + w.clone() * v.clone());
    
        if self.non_linear {
            sum.sigmoid()
        } else {
            sum
        }
    }
}

impl Gradients for Neuron {
    fn parameters(&self) -> Vec<Value> {
        let mut out = self.weights.clone();
        out.push(self.bias.clone());
        out
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(n_inputs: usize, n_outputs: usize, non_linear: bool) -> Self {
        let neurons = (0..n_outputs)
            .map(|_| Neuron::new(n_inputs, non_linear))
            .collect();
        Self { neurons }
    }

    pub fn forward(&self, val: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|x| x.forward(val)).collect()
    }
}

impl Gradients for Layer {
    fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}


pub struct MLP {
    size: Vec<usize>,
    layers: Vec<Layer>
}

impl MLP {
    pub fn new(n_inputs: usize, n_outputs: Vec<usize>) -> Self {
        let mut size = vec![n_inputs];
        size.extend(n_outputs);
        let layers = size.windows(2).enumerate().map(|(i, pair)| Layer::new(pair[0], pair[1], i != size.len() - 2)).collect();
        Self {size, layers }
    }

    pub fn forward(&self, val: &[Value]) -> Vec<Value> {
        let mut out = val.to_vec();
        for layer in &self.layers {
            out = layer.forward(&out);
        }
        out   
    }
}

impl Gradients for MLP {
    fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}
