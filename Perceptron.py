import tensorflow as tf

# Datos de entrenamiento
train_inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

train_labels = [
    [0],
    [1],
    [1],
    [0]
]

# Función para transformar las etiquetas en un formato binario
def transform_labels(labels):
    transformed_labels = []
    for label in labels:
        if label == 0:
            transformed_labels.append([1, 0])
        else:
            transformed_labels.append([0, 1])
    return transformed_labels

train_labels = transform_labels(train_labels)

# Modelo del perceptrón
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(2, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compila el modelo
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# Entrena el modelo
history = model.fit(train_inputs, train_labels, epochs=5000)

# Evalúa el modelo
predictions = model.predict(train_inputs)
predictions = [1 if p > 0.5 else 0 for p in predictions]

print("Predicciones:", predictions)
print("Precisión:", history.history['accuracy'][-1])