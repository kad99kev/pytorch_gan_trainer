import torch
import torch.nn as nn


class Generator(nn.Module):
    """Generator model for DCGAN."""

    def __init__(self, target_size, num_channels, latent_size, feature_size):
        """
        :param target_size: Target size of input image.
        :type target_size: int
        :param num_channels: Number of channels in images of dataset.
        :type num_channels: int
        :param latent_size: Size of input noise, defaults to 100.
        :param latent_size: int
        :param feature_size: Feature size of the model, defaults to 64.
        :type feature_size: int
        """
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

        self.input_layer = nn.Sequential(
            nn.ConvTranspose2d(latent_size, feature_size * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_size * 2),
            nn.ReLU(True),
        )
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

    def forward(self, inputs):
        """Forward pass to the model.

        :param inputs: Inputs to the model.
        :type inputs: torch.Tensor
        :returns: Outputs from the forward pass.
        """
        x = inputs.view(inputs.shape[0], inputs.shape[1], 1, 1)
        x = self.input_layer(x)
        x = self.conv_trans_1(x)
        x = self.conv_trans_2(x)
        x = self.conv_trans_3(x)
        if self.target_size >= 128:
            x = self.conv_trans_4(x)
            if self.target_size == 256:
                x = self.conv_trans_5(x)
        return self.output_layer(x)
