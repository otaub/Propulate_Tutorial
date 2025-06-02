import os
import random

from mpi4py import MPI
import torch
from torch import nn
from torchvision.datasets import MNIST

import propulate

from utils import get_dataloaders


GPUS_PER_NODE = int(os.environ["SLURM_GPUS_PER_NODE"])
NUM_WORKERS = int(os.environ["SLURM_CPUS_PER_TASK"])
checkpoint_path = "./"
dataset_path = f"/scratch/{os.environ['SLURM_JOB_ACCOUNT']}/{os.environ['USER']}/mnist"
device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
num_generations = 100
num_islands = 2
migration_prob = 0.1

seed = 42
pollination = True
limits = {
    "num_layers": (2, 10),
    "activation": ("relu", "sigmoid", "tanh"),
    "lr": (0.01, 0.0001),
    "d_hidden": (2, 128),
    "batch_size": ("1", "2", "4", "8", "16", "32", "64", "128"),
}

num_epochs = 32
# NOTE map categorical variable to python objects
activations = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
}


class CNN(nn.Module):
    def __init__(self, num_layers, activation, d_hidden):
        super().__init__()

        layers = []  # Set up the model architecture (depending on number of convolutional layers specified).
        layers += [
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=d_hidden, kernel_size=3, padding=1),
                activation(),
            ),
        ]
        layers += [
            nn.Sequential(
                nn.Conv2d(in_channels=d_hidden, out_channels=d_hidden, kernel_size=3, padding=1),
                activation(),
            )
            for _ in range(num_layers - 1)
        ]

        # NOTE due to padding output of final conv layer is 28*28*d_hidden
        self.fc = nn.Linear(in_features=28*28*d_hidden, out_features=10)
        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def loss_fn(params):
    num_layers = int(params["num_layers"])  # Number of convolutional layers
    activation = str(params["activation"])  # Activation function
    lr = float(params["lr"])  # Learning rate
    batch_size = int(params["batch_size"])
    d_hidden = int(params["d_hidden"])

    train_dl, val_dl = get_dataloaders(batch_size=batch_size, num_workers=NUM_WORKERS, root=dataset_path)

    crit = nn.CrossEntropyLoss()
    activation = activations[activation]
    
    net = CNN(num_layers, activation, d_hidden).to(device)

    best = 0.
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    for epoch in range(num_epochs):
        # NOTE train
        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = net(x)
            loss_val = crit(pred, y)
            loss_val.backward()
            optimizer.step()

        # NOTE eval
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device)
                y = y.to(device)
                pred = net(x)

                _, pred = torch.max(pred, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()
                acc = correct/total
            if  -acc < best:
                best = -acc

    # NOTE smaller is better
    return best


if __name__ == "__main__":
    # NOTE download dataset
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        MNIST(download=True, root=dataset_path, transform=None, train=True)
        MNIST(download=True, root=dataset_path, transform=None, train=False)
    comm.Barrier()

    rng = random.Random(seed + comm.rank)  # Set up separate random number generator for evolutionary optimizer.
    propagator = propulate.utils.get_default_propagator(pop_size=10, limits=limits, rng=rng)

    islands = propulate.Islands(
        loss_fn=loss_fn,
        propagator=propagator,
        rng=rng,
        generations=num_generations,
        num_islands=num_islands,
        migration_probability=migration_prob,
        pollination=pollination,
        checkpoint_path=checkpoint_path,
    )

    islands.propulate()
