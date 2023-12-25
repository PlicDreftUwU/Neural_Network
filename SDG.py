import numpy as np
import matplotlib.pyplot as plt

# Función cuadrática de ejemplo (simula la función de pérdida)
def quadratic_function(x):
    return 2 * x**2 + 3 * x + 1

# Derivada de la función cuadrática (gradiente)
def gradient(x):
    return 4 * x + 3

# Descenso de Gradiente Estocástico
def stochastic_gradient_descent(initial_guess, learning_rate, epochs):
    x_values = [initial_guess]
    for epoch in range(epochs):
        # Selecciona un ejemplo de entrenamiento aleatorio (simulando SGD)
        random_example = np.random.uniform(-5, 5)
        
        # Calcula el gradiente para el ejemplo seleccionado
        grad = gradient(random_example)
        
        # Actualiza el parámetro utilizando el descenso de gradiente
        updated_value = x_values[-1] - learning_rate * grad
        
        # Almacena el nuevo valor
        x_values.append(updated_value)
    return x_values

# Parámetros iniciales
initial_guess = np.random.uniform(-5, 5)
learning_rate = 0.1
epochs = 50

# Aplica Descenso de Gradiente Estocástico
trajectory = stochastic_gradient_descent(initial_guess, learning_rate, epochs)

# Visualización
plt.figure(figsize=(10, 6))
x = np.linspace(-6, 6, 100)
plt.plot(x, quadratic_function(x), label='Función de pérdida')
plt.scatter(trajectory, [quadratic_function(x) for x in trajectory], color='red', label='Descenso de Gradiente Estocástico')
plt.title('Descenso de Gradiente Estocástico en una función cuadrática')
plt.xlabel('Parámetro')
plt.ylabel('Valor de la función de pérdida')
plt.legend()
plt.show()
