import tensorflow as tf

# Enable eager execution for easier debugging
tf.executing_eagerly()

# Crear un grafo
graph = tf.Graph()

# Establecer el grafo actual como el grafo creado
with graph.as_default():
    # Crear variables, sesiones, operaciones y ejecutarlos
    a = tf.constant(5)
    b = tf.constant(3)
    add = tf.add(a, b)
    
    # Mostrar el resultado de la operación directamente
    print(add) # Muestra tf.Tensor(8, shape=(), dtype=int32)