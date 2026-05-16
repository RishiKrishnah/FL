from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

DEVICE = torch.device("cpu")


# MODEL
class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):

        x = x.view(-1, 28 * 28)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)


# DATASET
transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])

trainset = MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)


# LOAD CLIENT DATA
def load_data(partition_id, num_partitions):

    partition_size = len(trainset) // num_partitions

    start = partition_id * partition_size
    end = start + partition_size

    subset = torch.utils.data.Subset(
        trainset,
        list(range(start, end))
    )

    trainloader = DataLoader(
        subset,
        batch_size=32,
        shuffle=True,
    )

    return trainloader


# TRAIN FUNCTION
def train(net, trainloader):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=0.001,
    )

    net.train()

    for epoch in range(1):

        for images, labels in trainloader:

            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = net(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()


# TEST FUNCTION
def test(net, trainloader):

    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    loss = 0.0

    net.eval()

    with torch.no_grad():

        for images, labels in trainloader:

            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = net(images)

            loss += criterion(outputs, labels).item()

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return loss, accuracy


# GET PARAMETERS
def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


# SET PARAMETERS
def set_parameters(net, parameters):

    params_dict = zip(net.state_dict().keys(), parameters)

    state_dict = OrderedDict(
        {
            k: torch.tensor(v)
            for k, v in params_dict
        }
    )

    net.load_state_dict(state_dict, strict=True)