from torch.autograd import grad
from typing import List
from torch import Tensor
import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.set_grad_enabled(True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Define transformations for the training and test sets
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# Define the neural network
class SimpleANN(nn.Module):
    def __init__(self):
        super(SimpleANN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
model = SimpleANN().to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training function
def train(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# Testing function
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            labels = labels.to(device)
            outputs = model(inputs.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total}%")

# Train and test the model

def compute_sensitivities_wrt_input(
    model: torch.nn.Module, 
    x: Tensor
) -> List[Tensor]:
    """
    Function to get weight sensitivity wrt input for any and all models.
    """
    outputs = model(x)
    weights_model = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.split('.')[1] == "weight":
            print(name, param)
            weights_model.append(param)
    
    all_classes_grad = []
    for w_x in weights_model:
        grad_f = grad(
            outputs=outputs[0], 
            inputs=w_x, 
            grad_outputs=torch.ones_like(outputs[0]), 
            retain_graph=True
        )
        all_classes_grad.append(grad_f[0])  

    sensitivites = []
    for i, x in enumerate(weights_model):
        sensitivites.append(torch.mul(x, all_classes_grad[i]))

    print(len(sensitivites))
    return sensitivites

if __name__ == "__main__":
    # train(model, train_loader, criterion, optimizer, epochs=5)
    # test(model, test_loader)
    for x, y in train_dataset:
        save_x = x.to(device)
        save_y = y
        break
    
    print("Label is :", save_y)
    save_x.requires_grad = True
    
    all_grad = compute_sensitivities_wrt_input(model=model, x=save_x)
    # print(len(all_grad))
    print(all_grad)

