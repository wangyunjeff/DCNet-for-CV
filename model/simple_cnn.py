
import torch
from torch import nn

from Deformable_Convolutional_Networks.deformable_convolutional_networks import DeformableConv2d


class MNISTClassifier(nn.Module):
    def __init__(self,
                 deformable=False):
        super(MNISTClassifier, self).__init__()
        conv = nn.Conv2d if deformable == False else DeformableConv2d

        # self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=True)
        # self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv1 = conv(1, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = conv(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = conv(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv4 = conv(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = conv(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)  # [14, 14]
        x = torch.relu(self.conv2(x))
        x = self.pool(x)  # [7, 7]
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = self.gap(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x

