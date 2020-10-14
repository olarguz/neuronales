const tf = require("@tensorflow/tfjs");
const perceptron = require("../perceptron/perceptron");

console.log("Compuerta XOr");

let layers = {
  input: {
    units: 2,
    activation: "sigmoid",
  },
  output: {
    units: 1,
    activation: "sigmoid",
  },
};
model = perceptron.createModel(layers);

let matIn = [
  [1.0, 1.0],
  [1.0, 0.0],
  [0.0, 1.0],
  [0.0, 0.0],
];
let matOut = [[0.0], [1.0], [1.0], [0.0]];
const real = {
  inputs: tf.tensor2d(matIn),
  outputs: tf.tensor2d(matOut),
};
const test = {
  inputs: tf.tensor2d(matIn),
  outputs: tf.tensor2d(matOut),
};

const options = {
  epochs: 50000,
  batchSize: 4,
  shuffle: true,
  validationData: [test.inputs, test.outputs],
};
model.fit(real.inputs, real.outputs, options).then(() => {
  model.predict(tf.tensor2d([[1.0, 1.0]])).print();
  model.predict(tf.tensor2d([[1.0, 0.0]])).print();
  model.predict(tf.tensor2d([[0.0, 1.0]])).print();
  model.predict(tf.tensor2d([[0.0, 0.0]])).print();
});
