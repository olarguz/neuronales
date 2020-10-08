const tf = require("@tensorflow/tfjs");

exports.createModel = (layers) => {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: layers.input.units,
      inputShape: [layers.input.units],
      activation: layers.input.activation,
    })
  );
  if (layers.inner) {
    layers.inner.units.map((unit) =>
      model.add(
        tf.layers.dense({ units: unit, activation: layers.inner.activation })
      ));
  }
  model.add(
    tf.layers.dense({
      units: layers.output.units,
      kernelInitializer: "varianceScaling",
      activation: layers.output.activation,
    })
  );
  model.compile({
    loss: "meanAbsoluteError",
    optimizer: tf.train.sgd(0.025),
    metrics: ["accuracy"],
  });
  return model;
};
