// Simple 2-2-1 neural network (one hidden layer)

// Inputs
let x1 = 1.0, x2 = 2.0, y = 1.0;

// Learning rate
let lr = 0.1;

// Weights and biases
let w11 = 0.4, w21 = 0.3; // to hidden neuron 1
let w12 = 0.2, w22 = 0.5; // to hidden neuron 2
let b1 = 0.1, b2 = 0.2;   // hidden biases

let v1 = 0.6, v2 = 0.9;   // hidden → output
let b3 = 0.3;             // output bias

// Sigmoid activation
function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function sigmoidPrime(z) {
  const s = sigmoid(z);
  return s * (1 - s);
}

// ---- Forward pass ----
let z1 = w11 * x1 + w21 * x2 + b1;
let a1 = sigmoid(z1);

let z2 = w12 * x1 + w22 * x2 + b2;
let a2 = sigmoid(z2);

let z3 = v1 * a1 + v2 * a2 + b3;
let a3 = sigmoid(z3);

let loss = 0.5 * Math.pow(a3 - y, 2);
console.log("Initial loss:", loss.toFixed(4));

// ---- Backward pass ----
// Output layer gradients
let dL_da3 = a3 - y;
let da3_dz3 = sigmoidPrime(z3);

let dL_dv1 = dL_da3 * da3_dz3 * a1;
let dL_dv2 = dL_da3 * da3_dz3 * a2;
let dL_db3 = dL_da3 * da3_dz3;

// Hidden layer gradients
let dz3_da1 = v1;
let dz3_da2 = v2;

let dL_da1 = dL_da3 * da3_dz3 * dz3_da1;
let dL_da2 = dL_da3 * da3_dz3 * dz3_da2;

let da1_dz1 = sigmoidPrime(z1);
let da2_dz2 = sigmoidPrime(z2);

let dL_dz1 = dL_da1 * da1_dz1;
let dL_dz2 = dL_da2 * da2_dz2;

let dL_dw11 = dL_dz1 * x1;
let dL_dw21 = dL_dz1 * x2;
let dL_db1 = dL_dz1;

let dL_dw12 = dL_dz2 * x1;
let dL_dw22 = dL_dz2 * x2;
let dL_db2 = dL_dz2;

// ---- Update weights ----
w11 -= lr * dL_dw11;
w21 -= lr * dL_dw21;
b1 -= lr * dL_db1;

w12 -= lr * dL_dw12;
w22 -= lr * dL_dw22;
b2 -= lr * dL_db2;

v1 -= lr * dL_dv1;
v2 -= lr * dL_dv2;
b3 -= lr * dL_db3;

console.log("Updated weights:");
console.log({ w11, w21, w12, w22, v1, v2 });
console.log("Updated biases:");
console.log({ b1, b2, b3 });


