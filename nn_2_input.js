// Simple neural network: 2 inputs -> 1 output neuron

import { activationFunc, calculateLoss, getActivationFuncDerivative } from "./utils.js";

// Mock data
let x1 = 1.5;
let x2 = 2.0;
let y = 1.0;

// Initial parameters
let w1 = 0.4;
let w2 = 0.3;
let b = 0.2;
let lr = 0.1;  // learning rate

// Forward pass
let z = w1 * x1 + w2 * x2 + b; // weighted sum
let a = activationFunc(z);            // activation output

// Compute loss
let loss = calculateLoss(a, y)
console.log("Initial loss:", loss.toFixed(4));

// --- Backpropagation ---

// Step 1: Derivatives
let dL_da = a - y;           // ∂L/∂a
let da_dz = getActivationFuncDerivative(a); // ∂a/∂z

// Step 2: Chain rule for each parameter
let dz_dw1 = x1;
let dz_dw2 = x2;
let dz_db = 1;

let dL_dw1 = dL_da * da_dz * dz_dw1;
let dL_dw2 = dL_da * da_dz * dz_dw2;
let dL_db = dL_da * da_dz * dz_db;

console.log("Gradient of the loss w.r.t w1: ", dL_dw1.toFixed(4))
console.log("Gradient of the loss w.r.t w2: ", dL_dw2.toFixed(4))
console.log("Gradient of the bias:", dL_db.toFixed(4))

if (dL_dw1 < 0) {
  console.log("Gradient of the loss w.r.t w1 is negative, we increase w1 a little")
} else {
  console.log("Gradient of the loss w.r.t w1 is possitive, we decrease w1 a little")
}

if (dL_dw2 < 0) {
  console.log("Gradient of the loss w.r.t w2 is negative, we increase w2 a little")
} else {
  console.log("Gradient of the loss w.r.t w2 is possitive, we decrease w2 a little")
}

// Step 3: Gradient descent updates
w1 = w1 - lr * dL_dw1;
w2 = w2 - lr * dL_dw2;
b = b - lr * dL_db;

// Results
console.log("Updated weight w1:", w1.toFixed(4));
console.log("Updated weight w2:", w2.toFixed(4));
console.log("Updated bias b:", b.toFixed(4));
