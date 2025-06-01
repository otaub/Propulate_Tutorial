from typing import Tuple

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.datasets import MNIST


def get_dataloaders(batch_size: int, num_workers, root) -> Tuple[DataLoader, DataLoader]:
    """
    Get MNIST train and validation dataloaders.

    Parameters
    ----------
    batch_size : int
        The batch size.
    num_workers : int
        Number of dataloader workers

    Returns
    -------
    torch.utils.data.DataLoader
        The training dataloader.
    torch.utils.data.DataLoader
        The validation dataloader.
    """
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    # Note that the MNIST dataset has already been downloaded before globally by rank 0 in the main part.
    train_loader = DataLoader(
        dataset=MNIST(download=False, root=root, transform=data_transform, train=True),  # Use MNIST training dataset.
        batch_size=batch_size,  # Batch size
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        shuffle=True,  # Shuffle data.
    )
    val_loader = DataLoader(
        dataset=MNIST(download=False, root=root, transform=data_transform, train=False),  # Use MNIST testing dataset.
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        batch_size=1,  # Batch size
        shuffle=False,  # Do not shuffle data.
    )
    return train_loader, val_loader
