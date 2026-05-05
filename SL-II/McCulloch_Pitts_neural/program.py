import numpy as np

def mp_neuron(inputs, weights, threshold):
    weighted_sum = np.dot(inputs, weights)
    output = 1 if weighted_sum >= threshold else 0
    return output

def and_not(x1, x2):
    weights = [1, -1]
    threshold = 1
    inputs = np.array([x1, x2])
    output = mp_neuron(inputs, weights, threshold)
    return output

# Test cases
print(and_not(0, 0))  # Output: 0
print(and_not(1, 0))  # Output: 1
print(and_not(0, 1))  # Output: 0
print(and_not(1, 1))  # Output: 0


# Define the input and output vectors for the AND NOT function
X = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])
Y = np.array([0, 1, 0, 0])  # Output for AND NOT

# Initialize the weights and bias
w = np.array([0.0, 0.0], dtype=np.float64)
b = 0.0

# Set the learning rate and number of iterations
lr = 0.1
n_iter = 10

# Define the step function for the output
def step(x):
    return 1 if x > 0 else 0

# Train the network using the perceptron learning algorithm
for i in range(n_iter):
    for j in range(len(X)):
        x = X[j]
        y = Y[j]
        z = np.dot(x, w) + b
        o = step(z)
        dw = lr * (y - o) * x
        db = lr * (y - o)
        w += dw
        b += db

# Test the network using the trained weights and bias
test_x = np.array([1, 0])  # You can change this for different tests
test_z = np.dot(test_x, w) + b
test_o = step(test_z)

# Print the results
print("Input: ", test_x)
print("Output: ", test_o)
