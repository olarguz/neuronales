const tf = require("@tensorflow/tfjs");

exports.createModel = (layers) => {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: layers.input,
      inputShape: [layers.input],
      activacion: "sigmoid",
    })
  );
  layers.inner.map((inner) =>
    model.add(tf.layers.dense({ units: inner, activation: "sigmoid" }))
  );
  model.add(
    tf.layers.dense({
      units: layers.output,
      kernelInitializer: "varianceScaling",
      activation: "softmax",
    })
  );
  model.compile({
    loss: "meanSquaredError",
    optimizer: tf.train.adam(),
    metrics: ["accuracy"],
  });
  return model;
};
