import pytest
import torch
import pytorch_gan_trainer as pgt


def test_invalid_dataset():
    with pytest.raises(NotImplementedError) as _:
        pgt.datasets.prepare_dataloader("fail", 64, 64)


def test_load_from_path():
    dataloader, classes = pgt.datasets.prepare_dataloader("tests/test_images", 64, 1)
    assert type(dataloader) == torch.utils.data.dataloader.DataLoader
    assert type(classes) == list
    assert isinstance(len(classes), int)


def test_load_from_torch():
    dataloader, classes = pgt.datasets.prepare_dataloader("mnist", 64, 1)
    assert type(dataloader) == torch.utils.data.dataloader.DataLoader
    assert type(classes) == list
    assert isinstance(len(classes), int)
