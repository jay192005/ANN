import numpy as np
import matplotlib.pyplot as plt

# 1. Define the Activation Functions
def linear(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    # Softmax is usually applied to an array to get probabilities that sum to 1
    exp_x = np.exp(x - np.max(x)) # Subtracting max for numerical stability
    return exp_x / exp_x.sum()

# 2. Generate input data (x values from -10 to 10)
x = np.linspace(-10, 10, 100)

# 3. Setup the plotting area (2 rows, 3 columns)
plt.figure(figsize=(15, 10))
plt.suptitle("Neural Network Activation Functions", fontsize=16, fontweight='bold')

# Plot 1: Linear
plt.subplot(2, 3, 1)
plt.plot(x, linear(x), color='blue', linewidth=2)
plt.title("Linear Function\n$y = x$")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)

# Plot 2: Sigmoid
plt.subplot(2, 3, 2)
plt.plot(x, sigmoid(x), color='red', linewidth=2)
plt.title("Sigmoid Function\n$y = 1 / (1 + e^{-x})$")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)

# Plot 3: Tanh
plt.subplot(2, 3, 3)
plt.plot(x, tanh(x), color='green', linewidth=2)
plt.title("Tanh Function\n$y = \\tanh(x)$")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)

# Plot 4: ReLU
plt.subplot(2, 3, 4)
plt.plot(x, relu(x), color='purple', linewidth=2)
plt.title("ReLU Function\n$y = \max(0, x)$")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)

# Plot 5: Softmax
# Note: We use a smaller range for Softmax to visualize the probability distribution clearly
x_soft = np.linspace(-5, 5, 50) 
plt.subplot(2, 3, 5)
plt.plot(x_soft, softmax(x_soft), color='orange', linewidth=2)
plt.title("Softmax Function\n(Probability Distribution)")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)

# Adjust layout and show the plots
plt.tight_layout()
plt.subplots_adjust(top=0.9) # Adjust title spacing
plt.show()