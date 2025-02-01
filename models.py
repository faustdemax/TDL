import torch.nn as nn

class SimpleFCN(nn.Module):
    def __init__(self, input_size, width, output_size):
        super(SimpleFCN, self).__init__()
        self.fc1 = nn.Linear(input_size, width)
        self.fc2 = nn.Linear(width, output_size)
        self.activation = nn.Softplus(beta = 5)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x