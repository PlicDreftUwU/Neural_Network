import tensorflow as tf
from keras import layers, models

# Crear un modelo secuencial
model = models.Sequential()

# Agregar una capa densa con una neurona y una función de activación lineal
model.add(layers.Dense(units=1, input_shape=[1]))

# Compilar el modelo con el optimizador SGD y la función de pérdida MSE
model.compile(optimizer='sgd', loss='mean_squared_error')

# Datos de entrenamiento
x_train = [1.0, 2.0, 3.0, 4.0, 5.0]
y_train = [2.0, 4.0, 6.0, 8.0, 10.0]

# Entrenar el modelo
model.fit(x_train, y_train, epochs=1000)

# Hacer predicciones
x_test = [6.0, 7.0]
predictions = model.predict(x_test)
print("Predicciones:", predictions.flatten())
