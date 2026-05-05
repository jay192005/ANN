import numpy as np

class NeuralNet(object):
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, outputs, training_iterations):
        for _ in range(training_iterations):
            output = self.learn(inputs)
            error = outputs - output
            factor = np.dot(inputs.T, error * self.__sigmoid_derivative(output))
            self.synaptic_weights += factor

    def learn(self, inputs):
        return self.__sigmoid(np.dot(inputs, self.synaptic_weights))

if __name__ == "__main__":
    neural_network = NeuralNet()
    inputs = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 1]])
    outputs = np.array([[1, 0, 1]]).T
    neural_network.train(inputs, outputs, 10000)
    print(neural_network.learn(np.array([1, 0, 1])))




def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

input_size = 2
hidden_size = 2
output_size = 1

weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))

learning_rate = 0.1
num_epochs = 10000

for _ in range(num_epochs):
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_activation = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_activation, weights_hidden_output)
    output_layer_activation = sigmoid(output_layer_input)
    error = y - output_layer_activation
    output_delta = error * sigmoid_derivative(output_layer_activation)
    hidden_delta = output_delta.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_layer_activation)
    weights_hidden_output += hidden_layer_activation.T.dot(output_delta) * learning_rate
    weights_input_hidden += X.T.dot(hidden_delta) * learning_rate

hidden_layer_input = np.dot(X, weights_input_hidden)
hidden_layer_activation = sigmoid(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_activation, weights_hidden_output)
output_layer_activation = sigmoid(output_layer_input)

print(output_layer_activation)
