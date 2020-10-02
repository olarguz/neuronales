const tf = require("@tensorflow/tfjs");

const createModel = (input, inner, output) => {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: input,
      inputShape: [input],
      activacion: "relu",
    })
  );
  inner.map((layer) =>
    model.add(tf.layers.dense({ units: layer, activation: "sigmoid" }))
  );
  model.add(
    tf.layers.dense({
      units: output,
      kernelInitializer: "varianceScaling",
      activation: "softmax",
    })
  );
  model.compile({
    loss: "meanSquaredError",
    optimizer: "sgd",
    metrics: ["accuracy"],
  });
  return model;
};

exports.createModel = createModel;
