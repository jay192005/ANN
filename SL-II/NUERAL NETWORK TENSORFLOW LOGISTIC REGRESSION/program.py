import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the logistic regression model (single layer neural network)
class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.fc = nn.Linear(2, 1)  # 2 input features, 1 output (binary classification)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

# Dataset (XOR problem)
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])  # Input features
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])  # Target labels (binary)

# Initialize the model, loss function, and optimizer
model = LogisticRegressionModel()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Train the model
epochs = 10000
for epoch in range(epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate the model after training
with torch.no_grad():
    predicted = model(X)
    predicted = (predicted > 0.5).float()  # Convert probabilities to binary (0 or 1)
    accuracy = (predicted.eq(y).sum().item()) / y.size(0)  # Calculate accuracy
    print(f'Accuracy: {accuracy * 100:.2f}%')

# Print the final predictions
print("\nPredictions after training:")
print(predicted)
