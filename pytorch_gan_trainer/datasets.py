from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
import os


def prepare_dataloader(dataset_path, image_size, batch_size):
    """Prepares a PyTorch DataLoader from a given path \
        with a specific image and batch size. \
        If a specific dataset type is given, \
            the path will not be read and \
                a torchvision dataset will be loaded instead.

    Arguments:
        dataset_path (str): The location of the dataset.
        image_size (int): The image size of the images in the DataLoader.
        batch_size (int): The batch size to be set for the DataLoader.

    Returns:
        torch.utils.data.DataLoader: A DataLoader \
            with the specified image size and batch size, \
                Length of the number of classes found in the dataset.
    """

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5), std=(0.5)),
        ]
    )

    if _check_path(dataset_path):
        dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transform)
    else:
        dataset = _get_torch_dataset(dataset_path, transform)

    dataloader = DataLoader(
        dataset, batch_size, shuffle=True, pin_memory=True, num_workers=4
    )

    return dataloader, dataset.classes


def _check_path(path):
    """Checks whether the path given exists.

    Arguments:
        path (str): Path given.

    Returns:
        bool: True if the path exists, else False.
    """
    if os.path.exists(os.path.dirname(path)):
        return True


def _get_torch_dataset(d_type, transform):
    """Loads a torchvision dataset based on the name given.

    Arguments:
        d_type (str): Name of the dataset to be loaded.
        transform (torchvision.transforms.transforms.Compose): \
            The transformations to be carried out on the images.

    Returns
        torchvision.datasets: The PyTorch Dataset specified by d_type.
    """
    if d_type == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            "./dataset", train=True, download=True, transform=transform
        )

    elif d_type == "mnist":
        dataset = torchvision.datasets.MNIST(
            "./dataset", train=True, download=True, transform=transform
        )

    elif d_type == "fashion-mnist":
        dataset = torchvision.datasets.FashionMNIST(
            "./dataset", train=True, download=True, transform=transform
        )

    else:
        raise NotImplementedError()

    return dataset
