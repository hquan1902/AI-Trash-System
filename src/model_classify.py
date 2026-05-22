import torch
import torch.nn as nn
import torch.nn.functional as F


class TrashNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        # Convolution layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        # pooling
        self.pool = nn.MaxPool2d(2,2)

        # fully connected
        self.fc1 = nn.Linear(64*28*28, 128)
        self.fc2 = nn.Linear(128, num_classes)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x