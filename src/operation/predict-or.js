const tf = require("@tensorflow/tfjs");
require("tfjs-node-save");

const extractValue = (value) => Array.from(value.dataSync())[0];

const createInput = (input1, input2) => tf.tensor2d([[input1, input2]]);

const predict = (modelo, input1, input2) =>
  modelo.predict(createInput(input1, input2));

(async () => {
  console.log("Cargar el modelo Compuerta Or");
  let filename = "or-trained";
  const modelo = await tf.loadLayersModel(
    "file://./trained-models/".concat(filename).concat("/model.json")
  );

  console.log("Operar el modelo");
  console.log("And(1,1) =>", extractValue(predict(modelo, 1.0, 1.0)));
  console.log("And(1,0) =>", extractValue(predict(modelo, 1.0, 0.0)));
  console.log("And(0,1) =>", extractValue(predict(modelo, 0.0, 1.0)));
  console.log("And(0,0) =>", extractValue(predict(modelo, 0.0, 0.0)));
})();
