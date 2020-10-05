const tf = require("@tensorflow/tfjs");
const perceptron = require("../perceptron/perceptron");

console.log("Compuerta Or");

let layers = {
  input: 2,
  inner: [1, 2, 2],
  output: 1,
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
model
  .fit(real.inputs, real.outputs, {
    epochs: 120,
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
