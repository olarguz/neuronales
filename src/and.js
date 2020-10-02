const tf = require("@tensorflow/tfjs");

console.log("Compuerta And");

const height = tf.tensor2d([1, 1, 1, 0, 0, 1, 0, 0], [4, 2]);
const weight = tf.tensor2d([1, 0, 0, 0], [4, 1]);
const test = tf.ones([1,2]);

height.print();
weight.print();
test.print();

const model = tf.sequential();
model.add(
  tf.layers.dense({ units: 2, inputShape: [2], activacion: "sigmoid" })
);
model.add(tf.layers.dense({ units: 2, activation: "sigmoid" }));
model.add(tf.layers.dense({ units: 1 }));
model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

model
  .fit(height, weight, { epochs: 1000 })
  .then(() => {
    model.predict(tf.tensor2d([1,1],[1,2])).print();
  })
  .catch((err) => console.log(err));
