const tf = require("@tensorflow/tfjs");
const perceptron = require("../perceptron/perceptron");
const tools = require("../tools/tools");

console.log("Compuerta And");

let fileName = process.argv[2];

let layers = tools.readFile(fileName);
model = perceptron.createModel(layers);

let matIn = [
  [1.0, 1.0],
  [1.0, 0.0],
  [0.0, 1.0],
  [0.0, 0.0],
];
let matOut = [[1.0], [0.0], [0.0], [0.0]];
const real = {
  inputs: tf.tensor2d(matIn),
  outputs: tf.tensor2d(matOut),
};
const test = {
  inputs: tf.tensor2d(matIn),
  outputs: tf.tensor2d(matOut),
};

const options = {
  epochs: 15000,
  shuffle: true,
  validationData: [test.inputs, test.outputs],
};

model.fit(real.inputs, real.outputs, options).then(() => {
  model.predict(tf.tensor2d([[1.0, 1.0]])).print();
  model.predict(tf.tensor2d([[1.0, 0.0]])).print();
  model.predict(tf.tensor2d([[0.0, 1.0]])).print();
  model.predict(tf.tensor2d([[0.0, 0.0]])).print();
  console.log("-----------------------");
});
