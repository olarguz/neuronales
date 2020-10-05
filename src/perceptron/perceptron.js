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
  layers.inner.units.map((unit) =>
    model.add(
      tf.layers.dense({ units: unit, activation: layers.inner.activation })
    )
  );
  model.add(
    tf.layers.dense({
      units: layers.output.units,
      kernelInitializer: "varianceScaling",
      activation: layers.output.activation,
    })
  );
  model.compile({
    loss: "meanSquaredError",
    optimizer: tf.train.adam(),
    metrics: ["accuracy"],
  });
  return model;
};
