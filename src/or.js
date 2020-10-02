const tf = require("@tensorflow/tfjs");
const perceptron = require("./perceptron/perceptron");

console.log("Compuerta Or");

const height = tf.tensor2d([1, 1, 1, 0, 0, 1, 0, 0], [4, 2]);
const weight = tf.tensor2d([1, 1, 1, 0], [4, 1]);

height.print();
weight.print();

model = perceptron.createModel(2, [4], 1);

model.fit(height, weight, { epochs: 500 }).then(() => {
  model.predict(tf.tensor2d([0, 0], [1, 2])).print();
});
