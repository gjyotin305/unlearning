from torch.autograd import grad
from typing import List, Dict
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


def compute_sensitivities_wrt_input(
    model: torch.nn.Module, 
    x: Tensor
) -> List[Tensor]:
    """
    Function to get weight sensitivity wrt input for any and all models.
    """
    outputs = model(x)
    
    weights_model = get_model_weights(model=model)

    all_classes_grad = []
    for _, w_x in weights_model.items():
        grad_f = grad(
            outputs=outputs, 
            inputs=w_x, 
            grad_outputs=torch.ones_like(outputs), 
            retain_graph=True
        )
        all_classes_grad.append(grad_f[0])  

    sensitivites = []
    for i, weight in enumerate(weights_model.items()):
        _, w_x = weight
        sensitivites.append(torch.mul(w_x, all_classes_grad[i]))

    print(len(sensitivites))
    return sensitivites


def convertToMask(tensor_sample: Tensor) -> Tensor:
    z = torch.where(tensor_sample != -100, 1., 0.)
    return z


def top_k_sensi_tensor(sensi: Tensor, k: int) -> Tensor:
    sensi = torch.abs(sensi)
    l = int(k*sensi.shape[0]/100)
    print(l)
    desc = torch.argsort(-1*sensi, dim=1)
    indexes = []

    for x in desc:
        col_index = []
        for y in x:
            col_index.append(y)
            if len(col_index) == l:
                break
        indexes.append(col_index)

    sensi_copy = torch.clone(sensi)
    
    for i in range(len(indexes)):
        for j in indexes[i]:
            sensi_copy[i][j] = -100

    return sensi_copy


def get_final_masks(sensi: List[Tensor], k: int) -> List[Tensor]:
    top_k = []

    for x in sensi:
        top_k.append(top_k_sensi_tensor(sensi=x, k=k))
    
    top_k_masks = []
    for x in top_k:
        top_k_masks.append(convertToMask(x))

    return top_k_masks


def update_model_weights(model: nn.Module, masks: List[Tensor]) -> nn.Module:
    i = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.split('.')[1] == "weight":
            param.data = torch.multiply(masks[i], param.data)
            i += 1
    
    return model

def get_model_weights(model: torch.nn.Module) -> Dict[str, Tensor]:
    weights_model = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.split('.')[1] == "weight":
            weights_model[name.split('.')[0]] = param
    
    return weights_model


if __name__ == "__main__":
    train(model, train_loader, criterion, optimizer, epochs=5)
    test(model, test_loader)
    prev_checkpt = "mnist.pt"

    torch.save(model, prev_checkpt)

    for x, y in train_dataset:
        save_x = x.to(device)
        save_y = y
        break
    
    print("Label is :", save_y)
    save_x.requires_grad = True
    
    all_grad = compute_sensitivities_wrt_input(model=model, x=save_x)
    final_masks = get_final_masks(sensi=all_grad, k=30)
    model_new = update_model_weights(model=model, masks=final_masks)
    model_old = torch.load(prev_checkpt)
    
    test(model=model_old, test_loader=test_loader)
    test(model=model_new, test_loader=test_loader)

