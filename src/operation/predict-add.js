const { input } = require("@tensorflow/tfjs");
const tf = require("@tensorflow/tfjs");
require("tfjs-node-save");

const extractValue = (value) => Array.from(value.dataSync());

const createInput = (input1, input2) => tf.tensor2d([[input1, input2]]);

const predict = (modelo, input1, input2) =>
  modelo.predict(createInput(input1, input2));

(async () => {
  console.log("Cargar el modelo Compuerta And");
  let filename = "add-trained";
  const modelo = await tf.loadLayersModel(
    "file://./trained-models/".concat(filename).concat("/model.json")
  );

  console.log("Operar el modelo");
  console.log("Add(1,1) =>", extractValue(predict(modelo, 1.0, 1.0)));
  console.log("Add(10,4) =>", extractValue(predict(modelo, 10.0, 4.0)));
  console.log("Add(8,3) =>", extractValue(predict(modelo, 8.0, 3.0)));
  console.log("Add(0,4) =>", extractValue(predict(modelo, 0.0, 4.0)));
})();
