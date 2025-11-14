export const activationFunc = (z) => {
  return 1 / (1 + Math.exp(-z));
}

export const getActivationFuncDerivative = (a) => {
  const derivative = a * (1 - a)
  return derivative
}

export const calculateLoss = (a, y) => {
  return 0.5 * Math.pow(a - y, 2)
}