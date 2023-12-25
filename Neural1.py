import tensorflow as tf
from keras import layers, models
import numpy as np

# Datos de entrada y salida para el ejemplo
x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
y_train = np.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=float)

# Definición del modelo
model = models.Sequential([
    layers.Dense(units=1, input_shape=[1])  # Capa densa con 1 neurona y 1 entrada
])

# Compilación del modelo
model.compile(optimizer='sgd',  # Descenso de gradiente estocástico
              loss='mean_squared_error')  # Función de pérdida: error cuadrático medio

# Entrenamiento del modelo
model.fit(x_train, y_train, epochs=4000)  # Entrenar durante 1000 épocas

# Predicción con el modelo entrenado
x_test = np.array([6.0, 7.0], dtype=float)
predictions = model.predict(x_test)

print("Predicciones:", predictions.flatten())