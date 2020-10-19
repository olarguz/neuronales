# Ejercicio de Redes Neuronales con TensorFlow y Node.js

Este es un ejercicio para el manejo de redes neuronales usando TensorFlow y Node.js, el cual está separado en tres partes simples
que permiten su uso fácil.

- [X] Definición de un perceptrón de forma automática mediante un archivo de configuración.
- [X] Creación del proceso de entrenamiento.
- [ ] Uso del perceptrón entrenado.

## Archivo de configuración

El archivo de configuración consiste en un archivo en formato json, el cual consta de tres partes
- input: Corresponde a la capa de entradas y a la primera capa ocultas, la cual tiene el mismo numero en neuronas que la capa de entrada, esta primera capa oculta, lo mismo que la capa de entrada son obligatorias.
```
    "input": {
      "units": 2,
      "activation": "sigmoid"
    },
```
- inner: Correspondiente a las capas ocultas, esta es una sección opcional.
- output: Corresponde a la capa de salidas, la cual es obligatoria.
