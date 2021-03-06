# Ejercicio de Redes Neuronales con TensorFlow y Node.js

Este es un ejercicio para el manejo de redes neuronales usando TensorFlow y Node.js, el cual está separado en tres partes simples
que permiten su uso fácil.

- [X] Definición de un perceptrón de forma automática mediante un archivo de configuración.
- [X] Descripción de la estructura del perceptrón.
- [X] Creación del proceso de entrenamiento.
- [X] Uso del perceptrón entrenado.
- [X] Manual de ejecución.

## Archivo de configuración

El archivo de configuración consiste en un archivo en formato json, el cual consta de tres partes
- **`input`**: Corresponde a la capa de entradas y a la primera capa ocultas, la cual tiene el numero en neuronas de la capa de entrada definido en la etiqueta input, el numero de neuronas definido en la primera capa oculta se encuentra en la etiqueta units, lo mismo que la capa de entrada son obligatorias.
```
    "input": {
      "input": 2,
      "units": 4,
      "activation": "sigmoid"
    },
```
En el caso del ejemplo se aprecia que se crea una capa de entradas con dos neuronas y una capa oculta (obligatoria) de dos neuronas, junto con la funcion de activación sigmoide.
- **`inner`**: Correspondiente a las capas ocultas, esta es una sección opcional.
```
    "inner": {
      "units": [4],
      "activation": "sigmoid"
    },
```
En el caso del ejemplo se aprecia que se crea una capa de oculta con cuatro neuronas, junto con la funcion de activación sigmoide. Como la representacion de unidades corresponde a un arreglo, eso quier decir que se pueden agregar mas capas con número de neuronas variables.
- **`output`**: Corresponde a la capa de salidas, la cual es obligatoria.
```
    "output": {
      "units": 1,
      "activation": "sigmoid"
    }
```
En el caso del ejemplo se aprecia que se crea una capa de salida con una neurona, junto con la funcion de activación sigmoide.

El perceptron, con la configuración cargada se crea en con la funcion que se presenta a continuacion.
```
    let fileName = process.argv[2];
    let layers = tools.readFile(fileName);
    let modelo = perceptron.createModel(layers);
```
## Descripción de la estructura del perceptrón.

Iniciamos declarando un modelo secuencial
```
const model = tf.sequential();
```
Al cual se le asignan sus respectivas capas de entradas, capas ocultas y capa de salida, previamente configuradas, es importante mencionar que las capas deben tener el componente dense, para que exista una conexión entre una neurona de una capa con todas las neuronas de la siguiente capa

	
Capa de entrada, en units definimos cuantos nodos tendrá nuestra capa:
```	
tf.layers.dense({
  units: layers.input.units,
  inputShape: [layers.input.units],
  activation: layers.input.activation,
})
```
Capa oculta:
```
if (layers.inner) {
  layers.inner.units.map((unit) =>
    model.add(
      tf.layers.dense({ units: unit, activation: layers.inner.activation })
    )
  );
}
```
Capa de salida:
```
model.add(
  tf.layers.dense({
    units: layers.output.units,
    kernelInitializer: "varianceScaling",
    activation: layers.output.activation,
  })
);
```
### Descripción gráfica del operador AND
![Imagen Operador AND](./images/ejemploAND.png?raw=true "Operador AND")

## Proceso de entrenamiento.

El proceso de entrenamiento se lleva a cabo en la funcion trainning en la cual se toma el perceptrón, 
los datos con los cuales se desea entrenar el modelo y el nombre del archivo/carpeta en la cual se va
a almacenar el perceptron entrenado.

```
    let parameters = {
      fileName: "and-trained",
      epochs: 20000,
      shuffle: true,
    };

    await training(perceptron.createModel(layers), data, parameters);
```

### Almacenamiento del modelo entrenado.

Una vez que se entrena el modelo es importante se almacene, pues el proceso de entrenamiento es muy costoso por la cantidade de operaciones realizadas para el entrenamiento.
Para almacenar el modelo y los valores de los pesos en las conexiones entre las diferentes capas se utiliza la funcion *save* junto con el tipo de almacenamiento que se desea realizar, la ruta y el nombre de la carpeta en la que se va a almacernar la informacion.
Esta función produce dos archivos: *model.json* que tiene la estructura del perceptrón y *weights.bin* que tiene los valores de la red entrenada.
_Nota_: Es mandatorio el uso de la librería *tfjs-node-save*, pues ahí están definidos los handlers que permiten realizar el manejo del almacenamiento en el sistema de archivos. 

```
require("tfjs-node-save");


  modelo.save("file://./trained-models/".concat(fileName));

```

### Carga del modelo entrenado.

Con la red entrenada y el modelo almacenado en archivos ahora el proceso de uso del perceptrón entrenado para por la carga del perceptrón para poder ser usado en otra aplicación diferente de la aplicación de entrenamiento.
Para eso se puede usar la función *loadLayersModel*, la cual se encarga de cargar el modelo para su posterior uso.
_Nota_: Es mandatorio el uso de la librería *tfjs-node-save*, pues ahí están definidos los handlers que permiten realizar el manejo del almacenamiento en el sistema de archivos. 

```
require("tfjs-node-save");


  const modelo = await tf.loadLayersModel(
    "file://./trained-models/".concat(filename).concat("/model.json")
  );

```

## Uso del perceptron entrenado

Para el uso del perceptrón entrenado debemos primero cargarlo, luego empezar a usarlo.
En el ejemplo que se ha utilizado para ilustrar el proceso se requiere que el modelo reciba dos entrada que pueden tener los valores (1/0) y la respuesta es un solo valor que puede ser (1/0). 
En el caso de las redes neuronale el valor esperado es un aproximado al valor esperado. 
Con el entrenamiento realizado se espera que por lo menos el 95% certeza en los valores obtenidos, es decir cuando la respuesta debe ser 1, el perceptro responderá valores superiores a 0.95 y cuando el valor esperado es 0, se espera que los valores sean menores a 0.05.

```
  console.log("Operar el modelo");
  console.log("And(1,1) =>", extractValue(predict(modelo, 1.0, 1.0)));
  console.log("And(1,0) =>", extractValue(predict(modelo, 1.0, 0.0)));
  console.log("And(0,1) =>", extractValue(predict(modelo, 0.0, 1.0)));
  console.log("And(0,0) =>", extractValue(predict(modelo, 0.0, 0.0)));
```

## Manual de ejecución.

Para la ejeción del proyecto se apoya el uso del paquete de instalación de node npm.
- Para visualizar las operaciones existentes se ejecuta el siguiente comando:
```
npm run start
```
- Para ejecutar el entrenamiento se puede ejecutar el siguiente comando:
```
npm run training:[operacion] data/[operacion].json
```
- Para evaluar el perceptron entrenado se puede ejecutar el siguiente comando:
```
npm run predict:[operacion]
```
