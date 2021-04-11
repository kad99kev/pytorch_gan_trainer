import torch
import pytorch_gan_trainer as pgt


def test_import():
    print(torch.__version__)
    print(pgt)
    print(pgt.datasets)
    print(pgt.models)
