import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from models import SimpleFCN 
import matplotlib.pyplot as plt

def train_network(width, seed, train_loader, epochs=5):
    torch.manual_seed(seed)  # Ensure reproducibility
    model = SimpleFCN(input_size=28*28, width=width, output_size=10)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    return model