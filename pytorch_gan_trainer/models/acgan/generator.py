import torch
import torch.nn as nn


class Generator(nn.Module):
    """Generator model for ACGAN.

    Arguments:
        target_size (int): Target size of input image.
        num_channels (int): Number of channels in images of dataset.
        num_classes (int): Number of classes in dataset.
        latent_size (int): Size of input noise, defaults to 100.
        feature_size (int): Feature size of the model, defaults to 64.
    """

    def __init__(
        self, target_size, num_channels, num_classes, latent_size, feature_size
    ):
        super(Generator, self).__init__()

        assert target_size in [
            64,
            128,
            256,
        ], "Target size can only be one of the following: [64, 128, 256]"
        assert num_channels in [
            1,
            3,
        ], "Number of channels can only be one of the following: [1, 3]"

        self.target_size = target_size
        self.feature_size = feature_size

        # For labels
        self.label_embedding = nn.Embedding(num_classes, latent_size)
        self.linear = nn.Sequential(nn.Linear(latent_size, (feature_size * 2) * 4 ** 2))

        self.norm_layer = nn.Sequential(nn.BatchNorm2d(feature_size * 2), nn.ReLU(True))
        # Shape [4 x 4]

        # Conv Layers
        self.conv_trans_1 = nn.Sequential(
            nn.ConvTranspose2d(feature_size * 2, feature_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(True),
        )
        # Shape [8 x 8]

        self.conv_trans_2 = nn.Sequential(
            nn.ConvTranspose2d(feature_size, feature_size // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size // 2),
            nn.ReLU(True),
        )
        # Shape [16 x 16]

        self.conv_trans_3 = nn.Sequential(
            nn.ConvTranspose2d(
                feature_size // 2, feature_size // 2, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(feature_size // 2),
            nn.ReLU(True),
        )
        # Shape [32 x 32]
        feature_size = feature_size // 2

        # To adapt for higher image sizes
        if target_size >= 128:
            self.conv_trans_4 = nn.Sequential(
                nn.ConvTranspose2d(
                    feature_size, feature_size // 2, 4, 2, 1, bias=False
                ),
                nn.BatchNorm2d(feature_size // 2),
                nn.ReLU(True),
            )
            # Shape [64 x 64]
            feature_size = feature_size // 2

            if target_size == 256:
                self.conv_trans_5 = nn.Sequential(
                    nn.ConvTranspose2d(
                        feature_size, feature_size // 2, 4, 2, 1, bias=False
                    ),
                    nn.BatchNorm2d(feature_size // 2),
                    nn.ReLU(True),
                )
                # Shape [128 x 128]
                feature_size = feature_size // 2

        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(feature_size, num_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        # Shape [target_size x target_size]

    def forward(self, inputs, labels):
        """Forward pass to the model.

        Arguments:
            inputs (torch.Tensor): Inputs to the model.
            labels (torch.Tensor): Labels for conditional generation.

        Returns:
            torch.Tensor: Outputs from the forward pass.
        """
        x = torch.mul(self.label_embedding(labels), inputs)
        x = self.linear(x)
        x = x.view(x.shape[0], self.feature_size * 2, 4, 4)
        x = self.norm_layer(x)
        x = self.conv_trans_1(x)
        x = self.conv_trans_2(x)
        x = self.conv_trans_3(x)
        if self.target_size >= 128:
            x = self.conv_trans_4(x)
            if self.target_size == 256:
                x = self.conv_trans_5(x)
        return self.output_layer(x)
