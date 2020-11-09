const tf = require("@tensorflow/tfjs");

exports.training = async (modelo, data, fileName) => {
  const real = {
    inputs: tf.tensor2d(data.matIn),
    outputs: tf.tensor2d(data.matOut),
  };
  const test = {
    inputs: tf.tensor2d(data.matIn),
    outputs: tf.tensor2d(data.matOut),
  };

  const options = {
    epochs: 20000,
    batchSize: data.matIn.length,
    shuffle: true,
    validationData: [test.inputs, test.outputs],
  };

  await modelo.fit(real.inputs, real.outputs, options);
  modelo.save("file://./trained-models/".concat(fileName));
};
