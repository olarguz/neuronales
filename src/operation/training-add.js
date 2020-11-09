const tf = require("@tensorflow/tfjs");
require("tfjs-node-save");
const perceptron = require("../perceptron/perceptron");
const { training } = require("../perceptron/training");
const tools = require("../tools/tools");

const createAddInputData = (size, min, max) => {
  let delta = max - min;
  let inValues = [...Array(size).keys()].map((v) => [
    Math.random() * delta + min,
    Math.random() * delta + min,
  ]);
  let outValues = inValues.map((v) => [v[0] + v[1]]);
  return {
    matIn: inValues,
    matOut: outValues,
  };
};

(async () => {
  console.log("Entrenamiento Operacion ADD");
  let argv = process.argv;

  if (argv.length === 3) {
    let fileName = process.argv[2];
    let layers = tools.readFile(fileName);
    let data = createAddInputData(1000, -1000, 1000);

    await training(perceptron.createModel(layers), data, "add-trained");
  } else {
    console.error("Error: numero de parametros incorrectos");
    console.error("debe escribir el siguiente comando:");
    console.error("\t", "npm run training:[operation] [archivo.json]");
  }
})();

/*
Run with:
  npm run training:add data/add.json 
*/
