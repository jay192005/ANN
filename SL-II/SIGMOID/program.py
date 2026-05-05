import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Generate x values from -10 to 10
x = np.linspace(-10, 10, 100)
# Calculate sigmoid function values for x
y = sigmoid(x)
# Plot sigmoid function
plt.plot(x, y)
plt.title('Sigmoid Activation Function')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.show()