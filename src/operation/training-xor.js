const tf = require("@tensorflow/tfjs");
const perceptron = require("../perceptron/perceptron");
const tools = require("../tools/tools");

const training = async (modelo) => {
  let matIn = [
    [1.0, 1.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 0.0],
  ];
  let matOut = [[0.0], [1.0], [1.0], [0.0]];
  const real = {
    inputs: tf.tensor2d(matIn),
    outputs: tf.tensor2d(matOut),
  };
  const test = {
    inputs: tf.tensor2d(matIn),
    outputs: tf.tensor2d(matOut),
  };

  const options = {
    epochs: 50000,
    batchSize: 4,
    shuffle: true,
    validationData: [test.inputs, test.outputs],
  };
  await modelo.fit(real.inputs, real.outputs, options);

  await modelo.predict(tf.tensor2d([[1.0, 1.0]])).print();
  await modelo.predict(tf.tensor2d([[1.0, 0.0]])).print();
  await modelo.predict(tf.tensor2d([[0.0, 1.0]])).print();
  await modelo.predict(tf.tensor2d([[0.0, 0.0]])).print();
};

(() => {
  console.log("Entrenamiento Compuerta XOr");
  let argv = process.argv;

  if (argv.length === 3) {
    let fileName = process.argv[2];
    let layers = tools.readFile(fileName);

    await training(perceptron.createModel(layers));
  } else {
    console.error("Error: numero de parametros incorrectos");
    console.error("debe escribir el siguiente comando:");
    console.error("\t", "npm run training:[operation] [archivo.json]");
  }
})();
