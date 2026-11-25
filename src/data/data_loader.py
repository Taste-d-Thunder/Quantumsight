from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from configs.config import DATASETS


def get_data_loaders(dataset_name: str, batch_size: int, num_workers: int = 2):
    """Build train, validation, and test DataLoaders for a registered dataset.

    Args:
        dataset_name (str): Identifier from DATASETS registry ('MNIST', 'CIFAR100', 'STL10').
        batch_size (int): Batch size for DataLoader instances.
        num_workers (int): Subprocesses count for data loading.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
    """
    config = DATASETS[dataset_name]

    normalize_transform = None
    if 'transform' in config and hasattr(config['transform'], 'transforms'):
        for t in config['transform'].transforms:
            if isinstance(t, transforms.Normalize):
                normalize_transform = t
                break

    if dataset_name == 'STL10':
        raw_train = config['loader'](
            './data',
            split='train',
            download=True,
            transform=config['transform']
        )
        train_len = int(0.8 * len(raw_train))
        val_len = len(raw_train) - train_len
        train_data, val_data = random_split(raw_train, [train_len, val_len])

        test_transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(normalize_transform.mean, normalize_transform.std) if normalize_transform else transforms.Lambda(lambda x: x)
        ])
        test_data = config['loader'](
            './data',
            split='test',
            download=True,
            transform=test_transform
        )
    else:
        raw_train = config['loader'](
            './data',
            train=True,
            download=True,
            transform=config['transform']
        )
        train_len = int(0.8 * len(raw_train))
        val_len = len(raw_train) - train_len
        train_data, val_data = random_split(raw_train, [train_len, val_len])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normalize_transform.mean, normalize_transform.std) if normalize_transform else transforms.Lambda(lambda x: x)
        ])
        test_data = config['loader'](
            './data',
            train=False,
            download=True,
            transform=test_transform
        )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
