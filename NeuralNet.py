import torch
import torch.nn as nn
import torch.optim as optim
#Define the Neural Network Architecture: Define your neural network class by inheriting from nn.Module. For this example, let's create a simple feedforward neural network #with one hidden layer:

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
#In this example, input_size is the number of input features, hidden_size is the number of neurons in the hidden layer, and output_size is the number of output classes.

#Define Data and Labels: Prepare your training data and labels as PyTorch tensors. Here's a simple example:

# Sample data and labels (replace with your own data)
data = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
labels = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
#Initialize the Model, Loss Function, and Optimizer: Create an instance of your neural network, choose a loss function (e.g., Mean Squared Error for regression or Cross-#Entropy for classification), and select an optimizer (e.g., Stochastic Gradient Descent - SGD):

input_size = 2  # Number of input features
hidden_size = 4  # Number of neurons in the hidden layer
output_size = 1  # Number of output classes (1 for binary classification)

model = SimpleNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
optimizer = optim.SGD(model.parameters(), lr=0.1)  # Stochastic Gradient Descent
#Training Loop: Train the model by iterating through your data for multiple epochs:

num_epochs = 1000

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(data)
    loss = criterion(outputs, labels)

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
#Testing: After training, you can use the model to make predictions on new data:

with torch.no_grad():
    test_data = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0],[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    predictions = model(test_data)
    print(predictions)