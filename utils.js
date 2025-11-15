/**
 * Sigmoid Activation Function
 * Squashes any number to a value between 0 and 1
 * @param {number} z - The input value
 * @returns {number} - A value between 0 and 1
 */
export const sigmoid = (z) => {
  return 1.0 / (1.0 + Math.exp(-z));
};

/**
 * Derivative of the Sigmoid Function
 * Calculates how sensitive the sigmoid output is to changes in input
 * @param {number} z - The input value
 * @returns {number} - The rate of change at that point
 */
export const sigmoidDerivative = (z) => {
  const sigmoidValue = sigmoid(z);
  return sigmoidValue * (1 - sigmoidValue);
};

/**
 * Calculate Mean Squared Error loss
 * @param {number} predicted - The predicted value
 * @param {number} actual - The actual/correct value
 * @returns {number} - The loss value
 */
export const calculateLoss = (predicted, actual) => {
  return 0.5 * Math.pow(predicted - actual, 2);
};

/**
 * Generate a random number from a Gaussian (normal) distribution
 * with mean 0 and standard deviation 1
 * Uses the Box-Muller transform
 * @returns {number} - A random value from normal distribution
 */
export const randomGaussian = () => {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
};

/**
 * Shuffle an array in place using Fisher-Yates algorithm
 * @param {Array} array - The array to shuffle
 */
export const shuffleArray = (array) => {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
};