const tf = require("@tensorflow/tfjs");
require("tfjs-node-save");
const perceptron = require("../perceptron/perceptron");
const { training } = require("../perceptron/training");
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
