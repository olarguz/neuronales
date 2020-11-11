const tf = require("@tensorflow/tfjs");
require("tfjs-node-save");
const perceptron = require("../perceptron/perceptron");
const { training } = require("../perceptron/training");
const tools = require("../tools/tools");

const createOrInputData = () => {
  return {
    matIn: [
      [1.0, 1.0],
      [1.0, 0.0],
      [0.0, 1.0],
      [0.0, 0.0],
    ],
    matOut: [[1.0], [1.0], [1.0], [0.0]],
  };
};

(async () => {
  console.log("Entrenamiento Compuerta Or");
  let argv = process.argv;

  if (argv.length === 3) {
    let fileName = process.argv[2];
    let layers = tools.readFile(fileName);
    let data = createOrInputData();
    let parameters = {
      fileName: "or-trained",
      epochs: 20000,
      shuffle: true,
    };

    await training(perceptron.createModel(layers), data, parameters);
  } else {
    let error = "Error: numero de parametros incorrectos\n"
      .concat("debe escribir el siguiente comando:")
      .concat("\tnpm run training:[operation] [archivo.json]");
    console.error(error);
  }
})();
