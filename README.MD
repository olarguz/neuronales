# Ejercicio de Redes Neuronales con TensorFlow y Node.js

Este es un ejercicio para el manejo de redes neuronales usando TensorFlow y Node.js, el cual está separado en tres partes simples
que permiten su uso fácil.

- [X] Definición de un perceptrón de forma automática mediante un archivo de configuración.
- [X] Creación del proceso de entrenamiento.
- [X] Uso del perceptrón entrenado.
- [X] Manual de ejecución.

## Archivo de configuración

El archivo de configuración consiste en un archivo en formato json, el cual consta de tres partes
- input: Corresponde a la capa de entradas y a la primera capa ocultas, la cual tiene el numero en neuronas de la capa de entrada definido en la etiqueta input, el numero de neuronas definido en la primera capa oculta se encuentra en la etiqueta units, lo mismo que la capa de entrada son obligatorias.
```
    "input": {
      "input": 2,
      "units": 4,
      "activation": "sigmoid"
    },
```
En el caso del ejemplo se aprecia que se crea una capa de entradas con dos neuronas y una capa oculta (obligatoria) de dos neuronas, junto con la funcion de activación sigmoide.
- inner: Correspondiente a las capas ocultas, esta es una sección opcional.
```
    "inner": {
      "units": [4],
      "activation": "sigmoid"
    },
```
En el caso del ejemplo se aprecia que se crea una capa de oculta con cuatro neuronas, junto con la funcion de activación sigmoide. Como la representacion de unidades corresponde a un arreglo, eso quier decir que se pueden agregar mas capas con número de neuronas variables.
- output: Corresponde a la capa de salidas, la cual es obligatoria.
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

## Proceso de entrenamiento.

El proceso de entrenamiento se lleva a cabo en la funcion trainning en la cual se toma el perceptrón, 
los datos con los cuales se desea entrenar el modelo y el nombre del archivo/carpeta en la cual se va
a almacenar el perceptron entrenado.

```
    training(modelo,data,fileName);
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

_Pendiente_

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
