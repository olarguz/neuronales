const tf = require("@tensorflow/tfjs");
require("tfjs-node-save");
const perceptron = require("../perceptron/perceptron");
const tools = require("../tools/tools");

const createAndInputData = () => {
  return {
    matIn: [
      [1.0, 1.0],
      [1.0, 0.0],
      [0.0, 1.0],
      [0.0, 0.0],
    ],
    matOut: [[1.0], [0.0], [0.0], [0.0]],
  };
};

const training = async (modelo, data, fileName) => {
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
    shuffle: true,
    validationData: [test.inputs, test.outputs],
  };

  await modelo.fit(real.inputs, real.outputs, options);
  modelo.save("file://./trained-models/".concat(fileName));
};

(async () => {
  console.log("Entrenamiento Compuerta And");
  let argv = process.argv;

  if (argv.length === 3) {
    let fileName = process.argv[2];
    let layers = tools.readFile(fileName);
    let data = createAndInputData();

    await training(perceptron.createModel(layers), data, "and-trained");
  } else {
    console.error("Error: numero de parametros incorrectos");
    console.error("debe escribir el siguiente comando:");
    console.error("\t", "npm run training:[operation] [archivo.json]");
  }
})();
