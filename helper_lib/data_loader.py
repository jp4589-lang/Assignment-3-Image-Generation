import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

def _build_transforms(train: bool, augment: bool, normalize: bool):
    tfs = []
    if train and augment:
        tfs += [transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip()]
    tfs += [transforms.ToTensor()]
    if normalize:
        tfs += [transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD)]
    return transforms.Compose(tfs)

def get_data_loader(
    data_dir: str = "data",
    batch_size: int = 128,
    train: bool = True,
    augment: bool = True,
    normalize: bool = True,
    num_workers: int = 2,
    download: bool = True,
) -> DataLoader:
    """
    Returns a DataLoader for CIFAR-10 with optional augmentation/normalization.
    """
    tfm = _build_transforms(train=train, augment=augment, normalize=normalize)
    ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=train, transform=tfm, download=download
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader
