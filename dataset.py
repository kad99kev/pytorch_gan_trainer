from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
import os

def prepare_dataloader(dataset_path, image_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5))
    ])
    
    if _check_path(dataset_path):
        dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transform)
    else:
        dataset = _get_torch_dataset(dataset_path, transform)

    dataloader = DataLoader(dataset, batch_size, shuffle=True, pin_memory=True, num_workers=4)

    return dataloader, len(dataset.classes)

def _check_path(path):
    if os.path.exists(os.path.dirname(path)):
        return True

def _get_torch_dataset(d_type, transform):
    if d_type == 'cifar10':
        dataset = torchvision.datasets.CIFAR10('./dataset', train=True, download=True, transform=transform)
    
    elif d_type == 'mnist':
        dataset = torchvision.datasets.MNIST('./dataset', train=True, download=True, transform=transform)

    elif dtype == 'fashion-mnist':
        dataset = torchvision.datasets.FashionMNIST('./dataset', train=True, download=True, transform=transform)

    else:
        raise NotImplementedError

    return dataset

