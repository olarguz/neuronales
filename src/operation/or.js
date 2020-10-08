const tf = require("@tensorflow/tfjs");
const perceptron = require("../perceptron/perceptron");

console.log("Compuerta Or");

let layers = {
  input: {
    units: 2,
    activation: "sigmoid",
  },
  inner: {
    units: [1, 2, 2],
    activation: "sigmoid",
  },
  output: {
    units: 1,
    activation: "softmax",
  },
};
model = perceptron.createModel(layers);

const real = {
  inputs: tf.tensor2d([
    [1.0, 1.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 0.0],
  ]),
  outputs: tf.tensor1d([1.0, 1.0, 1.0, 0.0]),
};
const test = {
  inputs: tf.tensor2d([
    [1.0, 1.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 0.0],
  ]),
  outputs: tf.tensor1d([1.0, 1.0, 1.0, 0.0]),
};
const options = {
  epochs: 10,
  batchSize: 4,
  shuffle: true,
  validationData: [test.inputs, test.outputs],
};
model
  .fit(real.inputs, real.outputs, options)
  .then(() => {
    model.predict(tf.tensor2d([[1.0, 1.0]])).print();
    model.predict(tf.tensor2d([[1.0, 0.0]])).print();
    model.predict(tf.tensor2d([[0.0, 1.0]])).print();
    model.predict(tf.tensor2d([[0.0, 0.0]])).print();
  });
