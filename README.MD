# Ejercicio de Redes Neuronales con TensorFlow y Node.js

Este es un ejercicio para el manejo de redes neuronales usando TensorFlow y Node.js, el cual está separado en tres partes simples
que permiten su uso fácil.

- [X] Definición de un perceptrón de forma automática mediante un archivo de configuración.
- [X] Creación del proceso de entrenamiento.
- [ ] Uso del perceptrón entrenado.
- [X] Manual de ejecución.

## Archivo de configuración

El archivo de configuración consiste en un archivo en formato json, el cual consta de tres partes
- input: Corresponde a la capa de entradas y a la primera capa ocultas, la cual tiene el mismo numero en neuronas que la capa de entrada, esta primera capa oculta, lo mismo que la capa de entrada son obligatorias.
```
    "input": {
      "units": 2,
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

El proceso de entrenamiento se lleva a cabo en la funcion trainning en la cual se toma el perceptrón.

```
    training(modelo);
```

## Uso del perceptron entrenado

__Pendiente__

## Manual de ejecución.

Para la ejeción del lproyecto se apoya el uso del paquete de instalación de node npm.
- Para visualizar las operaciones existentes se ejecuta el siguiente comando:
```
npm run training:[operacion] data/[operacion].json
```
- Para ejecutar el entrenamiento se puede ejecutar el siguiente comando:
```
npm run start
```
