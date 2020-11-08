const tf = require("@tensorflow/tfjs");
require("tfjs-node-save");
const perceptron = require("../perceptron/perceptron");
const tools = require("../tools/tools");

const createAddInputData = (size) => {
  let inValues = [...Array(size).keys()].map((v) => [
    Math.random() * 1000,
    Math.random() * 1000,
  ]);
  let outValues = inValues.map((v) => [v[0] + v[1]]);
  return {
    matIn: inValues,
    matOut: outValues,
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
    batchSize: data.matIn.length,
    shuffle: true,
    validationData: [test.inputs, test.outputs],
  };
  await modelo.fit(real.inputs, real.outputs, options);
  modelo.save("file://./trained-models/".concat(fileName));
};

(async () => {
  console.log("Entrenamiento Operacion ADD");
  let argv = process.argv;

  if (argv.length === 3) {
    let fileName = process.argv[2];
    let layers = tools.readFile(fileName);
    let data = createAddInputData(1000);

    await training(perceptron.createModel(layers), data, "add-trained");
  } else {
    console.error("Error: numero de parametros incorrectos");
    console.error("debe escribir el siguiente comando:");
    console.error("\t", "npm run training:[operation] [archivo.json]");
  }
})();
