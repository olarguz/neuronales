const { layers } = require("@tensorflow/tfjs");
const tf = require("@tensorflow/tfjs");

console.log("Compuerta And");

const createModel = (input, inner, output) => {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: input,
      inputShape: [input],
      activacion: "sigmoid",
    })
  );
  inner.map((layer) =>
    model.add(tf.layers.dense({ units: layer, activation: "sigmoid" }))
  );
  model.add(tf.layers.dense({ units: output }));
  model.compile({ loss: "meanSquaredError", optimizer: "sgd" });
  return model;
};


const height = tf.tensor2d([1, 1, 1, 0, 0, 1, 0, 0], [4, 2]);
const weight = tf.tensor2d([1, 0, 0, 0], [4, 1]);
const test = tf.ones([1, 2]);

height.print();
weight.print();
test.print();

model = createModel(2, [4,3,4], 1);

model
  .fit(height, weight, { epochs: 1000 })
  .then(() => {
    model.predict(tf.tensor2d([1, 1], [1, 2])).print();
  })
  .catch((err) => console.log(err));
