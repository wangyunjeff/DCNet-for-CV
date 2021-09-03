import torch
import torchvision.ops
from torch import nn
import numpy as np
import random

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch
from torch import nn

from model.simple_cnn import MNISTClassifier

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)



def train(model, loss_function, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()


def test(model, loss_function, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    num_data = 0
    with torch.no_grad():
        for data, target in test_loader:
            org_data, target = data.to(device), target.to(device)

            for scale in np.arange(0.5, 1.6, 0.1):  # [0.5, 0.6, ... ,1.2, 1.3, 1.4, 1.5]
                data = transforms.functional.affine(org_data, scale=scale, angle=0, translate=[0, 0], shear=0)
                output = model(data)
                test_loss += loss_function(output, target).item()  # sum up batch mean loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                num_data += len(data)

    test_loss /= num_data

    test_acc = 100. * correct / num_data
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, num_data,
        test_acc))
    return test_acc


def main(use_deformable_conv=False):
    # Training settings
    seed = 1
    setup_seed(seed)

    use_cuda = torch.cuda.is_available()
    batch_size = 16
    lr = 1e-3
    gamma = 0.7
    epochs = 14

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 0,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_transform = transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.MNIST('./data', train=True, download=True,
                              transform=train_transform)
    dataset2 = datasets.MNIST('./data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = MNISTClassifier(use_deformable_conv).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    loss_function = nn.CrossEntropyLoss()
    best_test_acc = 0.
    for epoch in range(1, epochs + 1):
        train(model, loss_function, device, train_loader, optimizer, epoch)
        best_test_acc = max(best_test_acc, test(model, loss_function, device, test_loader))
        scheduler.step()

        print(epoch, "best top1 acc(%): ", f"{best_test_acc:.2f}")


main(use_deformable_conv=True)