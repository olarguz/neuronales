const tf = require("@tensorflow/tfjs");

exports.training = async (modelo, data, parameters) => {
  const real = {
    inputs: tf.tensor2d(data.matIn),
    outputs: tf.tensor2d(data.matOut),
  };
  const test = {
    inputs: tf.tensor2d(data.matIn),
    outputs: tf.tensor2d(data.matOut),
  };

  const options = {
    epochs: parameters.epochs,
    batchSize: data.matIn.length,
    shuffle: parameters.shuffle,
    validationData: [test.inputs, test.outputs],
  };

  await modelo.fit(real.inputs, real.outputs, options);
  modelo.save("file://./trained-models/".concat(parameters.fileName));
};
