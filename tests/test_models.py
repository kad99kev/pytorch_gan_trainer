import torch
import pytorch_gan_trainer as pgt


def test_ac_model_creation():
    ac_model = pgt.models.ACGAN(64, 3, 2)
    assert type(ac_model) == pgt.models.ACGAN


def test_dc_model_creation():
    dc_model = pgt.models.DCGAN(64, 3, 2)
    assert type(dc_model) == pgt.models.DCGAN


def test_ac_model_train():
    dataloader, classes = pgt.datasets.prepare_dataloader("tests/test_images", 64, 1)
    ac_model = pgt.models.ACGAN(64, 3, 2)
    ac_model.train(1, dataloader, output_path="test_outputs")
    out = ac_model.generate(torch.arange(0, len(classes)))
    assert type(out) == torch.Tensor


def test_dc_model_train():
    dataloader, classes = pgt.datasets.prepare_dataloader("tests/test_images", 64, 1)
    dc_model = pgt.models.DCGAN(64, 3, 2)
    dc_model.train(1, dataloader, output_path="test_outputs")
    out = dc_model.generate(1)
    assert type(out) == torch.Tensor
