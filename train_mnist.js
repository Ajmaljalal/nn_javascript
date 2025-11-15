import mnist from 'mnist';
import NeuralNetwork from './nn_hand_written_digits.js';

console.log('Loading MNIST dataset...');

// Load the MNIST dataset
// This gives us 60,000 training images and 10,000 test images
const mnistSet = mnist.set(8000, 2000); // 8000 training, 2000 test (for faster training)

// Format the data for our neural network
const trainingData = mnistSet.training.map(item => ({
  x: item.input,   // 784 pixel values (28x28 image flattened)
  y: item.output   // 10-element one-hot encoded array
}));

const testData = mnistSet.test.map(item => ({
  x: item.input,
  y: item.output
}));

console.log(`Training data: ${trainingData.length} images`);
console.log(`Test data: ${testData.length} images`);
console.log('');

// Create the neural network
// 784 inputs (28x28 pixels), 30 hidden neurons, 10 outputs (digits 0-9)
const network = new NeuralNetwork([784, 30, 10]);

console.log('Training neural network on MNIST dataset...');
console.log('This may take a few minutes...');
console.log('');

// Train the network
network.stochasticGradientDescent(
  trainingData,
  30,      // epochs - how many times to go through all training data
  10,      // miniBatchSize - update weights after every 10 examples
  0.5,     // learningRate - how big of adjustment steps to take
  testData // test data to evaluate performance after each epoch
);

console.log('');
console.log('Training complete! 🎉');

