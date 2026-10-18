use std::{cell::RefCell, ops::{Add, Mul, Neg, Div, Sub}, rc::Rc, collections::HashSet};

enum Op {
    Me,
    Add,
    Mul,
    Pow(f32),
    Relu,
    Neg,
    Sub,
    Div,
    Sigmoid,
}

#[derive(Clone)]
pub struct Value(Rc<RefCell<Node>>);

struct Node {
    data: f32,

    // Me set these as internal stuff
    // I'm stealing from micrograd
    grad: f32,
    op: Op,
    children: Vec<Value>
}

impl Value {
    pub fn new(data: f32) -> Self {
        Self::with_op(data, Op::Me, vec![])
    }

    pub fn with_op(data: f32, op: Op, children: Vec<Value>) -> Self {
        Value(Rc::new(RefCell::new(Node { data, grad: 0.0, op, children})))
    }

    pub fn zero_grad(&self) {
        self.0.borrow_mut().grad = 0.0;
    }

    pub fn set_data(&self, data: f32) {
        self.0.borrow_mut().data = data;
    }

    pub fn data(&self) -> f32 {
        self.0.borrow().data
    }

    pub fn grad(&self) -> f32 {
        self.0.borrow().grad
    }

    // Part of backward propagation
    fn step(&self) {
        let node = self.0.borrow();
        let out_grad = node.grad;
        match node.op {
            // basically when I create a node not the script
            Op::Me => {},
            
            Op::Mul => {
                let curr = &node.children[0];
                let prev = &node.children[1];

    
                let curr_data = curr.data();
                let prev_data = prev.data();
                curr.0.borrow_mut().grad += prev_data * out_grad;
                prev.0.borrow_mut().grad += curr_data * out_grad
            }
            Op::Add => {
                for child in &node.children {
                    child.0.borrow_mut().grad += out_grad;
                }
            }
            Op::Pow(exp) => {
                let curr = &node.children[0];

                let curr_data = curr.data();
                curr.0.borrow_mut().grad += (exp * curr_data.powf(exp - 1.0)) * out_grad
            }
            Op::Relu => {
                let curr = &node.children[0];
                curr.0.borrow_mut().grad += (node.data > 0.0) as i32 as f32 * out_grad
            },
            Op::Div => {}
            Op::Neg => {}
            Op::Sub => {},
            Op::Sigmoid => {
                let curr = &node.children[0];
                curr.0.borrow_mut().grad += node.data * (1.0 - node.data) * out_grad;
            }
        }
    } 

    pub fn pow(&self, n: f32) -> Value {
        Value::with_op(self.data().powf(n), Op::Pow(n), vec![self.clone()])
    }

    pub fn relu(&self) -> Value {
        Value::with_op(self.data().max(0.0), Op::Relu, vec![self.clone()])
    }

    

    pub fn sigmoid(&self) -> Value {
        let s = 1.0 / (1.0 + (-self.data()).exp());
        Value::with_op(s, Op::Sigmoid, vec![self.clone()])
    }
    
    fn build(v: &Value, visited: &mut HashSet<*const RefCell<Node>>, topology: &mut Vec<Value>) {
        if visited.insert(Rc::as_ptr(&v.0)) {
            for child in v.0.borrow().children.iter() {
                Self::build(child, visited, topology);
            }
            topology.push(v.clone());
        }
    }
    

    pub fn backward(&self) {
        let mut topology = vec![];
        let mut visited = HashSet::new();
        Self::build(self, &mut visited, &mut topology);
        self.0.borrow_mut().grad = 1.0;
        for v in topology.iter().rev() {
            v.step();
        }
    }
    
}


// Forward propagation
impl Add for Value {
    type Output = Value;
    fn add(self, rhs: Value) -> Value {
        let data = self.data() + rhs.data();
        Value::with_op(data, Op::Add, vec![self, rhs])
    }
}
 
impl Mul for Value {
    type Output = Value;
    fn mul(self, rhs: Value) -> Value {
        let data = self.data() * rhs.data();
        Value::with_op(data, Op::Mul, vec![self, rhs])
    }
}


impl Mul<f32> for Value {
    type Output = Value;
    fn mul(self, rhs: f32) -> Value { 
        self * Value::new(rhs) 
    }
}

impl Add<f32> for Value {
    type Output = Value;
    fn add(self, rhs: f32) -> Self::Output {
        self + Value::new(rhs)
    }
}

impl Mul<Value> for f32 {
    type Output = Value;
    fn mul(self, rhs: Value) -> Value {
        Value::new(self) * rhs
    }
}

impl Add<Value> for f32 {
    type Output = Value;
    fn add(self, rhs: Value) -> Value {
        Value::new(self) + rhs
    }
}

impl Neg for Value {
    type Output = Value;
    fn neg(self) -> Value { self * Value::new(-1.0) }
}

impl Sub for Value {
    type Output = Value;
    fn sub(self, rhs: Value) -> Value { self + (-rhs) }
}

impl Div for Value {
    type Output = Value;
    fn div(self, rhs: Value) -> Value { self * rhs.pow(-1.0) }
}

impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let n = self.0.borrow();
        write!(f, "Value(data={}, grad={})", n.data, n.grad)
    }
}