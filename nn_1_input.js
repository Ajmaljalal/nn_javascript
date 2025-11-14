import { activationFunc, calculateLoss, getActivationFuncDerivative } from "./utils.js"

const x = 2
const y = 3

let w = 4
let b = 0.5

const learning_rate = 0.2

// --- Forward pass ---

// wieghted sum of the input plus the bias
const z = x * w + b
// the activation function value
const a = activationFunc(z)
// calculate the loss of the network
const loss = calculateLoss(x, y)

console.log("Initial loss:", loss.toFixed(4));


// --- Backpropagation ---

// Derivatives using chain rule
const dL_da = a - y                             // ∂L/∂a
const da_dz = getActivationFuncDerivative(a)    // ∂a/∂z
const dz_dw = x                                 // ∂z/∂w
const dz_db = 1                                 // ∂z/∂b


// Chain rule: ∂L/∂w = ∂L/∂a * ∂a/∂z * ∂z/∂w
let dL_dw = dL_da * da_dz * dz_dw;
let dL_db = dL_da * da_dz * dz_db;

console.log("Gradient of the loss: ", dL_dw.toFixed(4))
console.log("Gradient of the bias:", dL_db.toFixed(4))

if (dL_dw < 0) {
  console.log("Gradient of the loss w.r.t w is negative, we increase w a little")
} else {
  console.log("Gradient of the loss w.r.t w is possitive, we decrease w a little")
}




// Gradient descent update

w = w - learning_rate * dL_dw
b = b - learning_rate * dL_db

console.log("Updated weight:", w.toFixed(4));
console.log("Updated bias:", b.toFixed(4));