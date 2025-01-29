import torch.nn as nn

class FullyConnectedNetwork(nn.Module):
    
    def __init__(self, input_size, width, output_size):
        super(FullyConnectedNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x    