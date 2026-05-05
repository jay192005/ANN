import numpy as np

j = int(input("Enter a Number (0-9): "))

step_function = lambda x: 1 if x >= 0 else 0

training_data = [
    {'input': [0, 0, 0, 0, 0, 0], 'label': 1},
    {'input': [0, 0, 0, 0, 0, 1], 'label': 0},
    {'input': [0, 0, 0, 0, 1, 0], 'label': 1},
    {'input': [0, 0, 0, 0, 1, 1], 'label': 0},
    {'input': [0, 0, 0, 1, 0, 0], 'label': 1},
    {'input': [0, 0, 0, 1, 0, 1], 'label': 0},
    {'input': [0, 0, 0, 1, 1, 0], 'label': 1},
    {'input': [0, 0, 0, 1, 1, 1], 'label': 0},
    {'input': [0, 0, 1, 0, 0, 0], 'label': 1},
    {'input': [0, 0, 1, 0, 0, 1], 'label': 0},
]

weights = np.array([0, 0, 0, 0, 0, 0], dtype=int)

for data in training_data:
    x = np.array(data['input'])
    y = data['label']
    output = step_function(np.dot(x, weights))
    error = y - output
    weights += x * error

binary_input = np.array([int(bit) for bit in '{0:06b}'.format(j)])
prediction = step_function(np.dot(binary_input, weights))
parity = "even" if prediction == 1 else "odd"

print(j, "is", parity)

