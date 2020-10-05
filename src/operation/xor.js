const tf = require("@tensorflow/tfjs");
const perceptron = require("../perceptron/perceptron");

console.log("Compuerta XOr");

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
  outputs: tf.tensor1d([0.0, 1.0, 1.0, 0.0]),
};
const test = {
  inputs: tf.tensor2d([
    [1.0, 1.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 0.0],
  ]),
  outputs: tf.tensor1d([0.0, 1.0, 1.0, 0.0]),
};
model
  .fit(real.inputs, real.outputs, {
    epochs: 200,
    batchSize: 4,
    shuffle: true,
    validationData: [test.inputs, test.outputs],
  })
  .then(() => {
    model.predict(tf.tensor2d([[1.0, 1.0]])).print();
    model.predict(tf.tensor2d([[1.0, 0.0]])).print();
    model.predict(tf.tensor2d([[0.0, 1.0]])).print();
    model.predict(tf.tensor2d([[0.0, 0.0]])).print();
  });
