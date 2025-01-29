import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from models import FullyConnectedNetwork   
import matplotlib.pyplot as plt

def train_network(width, seed, train_loader, test_loader):
    torch.manual_seed(seed)

    model = FullyConnectedNetwork(input_size=784,width= width, output_size=10)
    model = model.to('cpu')

    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(5):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(data.size(0),-1)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return model


def plot_fluctuations_vs_width(results):
    widths = [r['width'] for r in results]
    fluctuations = [r['fluctuations'] for r in results]

    plt.figure(figsize=(8, 6))
    plt.loglog(widths, fluctuations, marker='o', label=r'$\delta F$ (Fluctuations)')
    plt.xlabel("Width (h)", fontsize=14)
    plt.ylabel(r"$\delta F$ (Fluctuations)", fontsize=14)
    plt.title("Fluctuations vs. Width", fontsize=16)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=12)
    plt.show()

def plot_test_error_vs_width(results):
    widths = [r['width'] for r in results]
    test_errors = [r['test_error'] for r in results]

    plt.figure(figsize=(8, 6))
    plt.plot(widths, test_errors, marker='o', label="Test Error", color='red')
    plt.xlabel("Width (h)", fontsize=14)
    plt.ylabel("Test Error", fontsize=14)
    plt.title("Test Error vs. Width", fontsize=16)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(fontsize=12)
    plt.show()