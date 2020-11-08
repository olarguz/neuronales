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

let data = createAddInputData(1000);
/*let inValues = [...Array(10).keys()].map((v) => [
  Math.random() * 1000,
  Math.random() * 1000,
]);
let outValues = inValues.map((v) => [v[0] + v[1]]);*/
console.log(data);
/*console.log(inValues);
console.log(outValues);*/
