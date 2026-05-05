import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

input_size = 2
hidden_size = 4
output_size = 1

np.random.seed(1)
W1 = 2 * np.random.rand(input_size, hidden_size) - 1
b1 = np.zeros((1, hidden_size))
W2 = 2 * np.random.rand(hidden_size, output_size) - 1
b2 = np.zeros((1, output_size))

lr = 0.5
epochs = 10000

for epoch in range(epochs):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    error = y - a2
    d_a2 = error * sigmoid_derivative(a2)

    d_W2 = np.dot(a1.T, d_a2)
    d_b2 = np.sum(d_a2, axis=0, keepdims=True)

    d_a1 = np.dot(d_a2, W2.T) * sigmoid_derivative(a1)

    d_W1 = np.dot(X.T, d_a1)
    d_b1 = np.sum(d_a1, axis=0, keepdims=True)

    W2 += lr * d_W2
    b2 += lr * d_b2
    W1 += lr * d_W1
    b1 += lr * d_b1

z1 = np.dot(X, W1) + b1
a1 = sigmoid(z1)
z2 = np.dot(a1, W2) + b2
a2 = sigmoid(z2)

print("Predictions:")
print(np.round(a2, 3))
