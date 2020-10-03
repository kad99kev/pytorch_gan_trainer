import torch
import torch.nn as nn

class Generator(nn.Module):
    '''
    Generator model for DCGAN.

    Attributes:
        target_size (int): Target size of output image.
        latent_size (int): Size of input noise, defaults to 100.
        feature_size (int): Feature size of the model, defaults to 64. 
    '''

    def __init__(self, target_size, latent_size, feature_size):
        super(Generator, self).__init__()

        assert target_size in [64, 128, 256], 'Target size can only be one of the following: [64, 128, 256]'

        self.target_size = target_size
        self.feature_size = feature_size

        self.input_layer = nn.Sequential(
            nn.ConvTranspose2d(latent_size, feature_size * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_size * 2),
            nn.ReLU(True)
        )
        # Shape [4 x 4]

        # Conv Layers
        self.conv_trans_1 = nn.Sequential(
            nn.ConvTranspose2d(feature_size * 2, feature_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(True)
        )
        # Shape [8 x 8]

        self.conv_trans_2 = nn.Sequential(
            nn.ConvTranspose2d(feature_size, feature_size // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size // 2),
            nn.ReLU(True),
        )
        # Shape [16 x 16]

        self.conv_trans_3 = nn.Sequential(
            nn.ConvTranspose2d(feature_size // 2, feature_size // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size // 2),
            nn.ReLU(True),
        )
        # Shape [32 x 32]
        feature_size = feature_size // 2

        # To adapt for higher image sizes
        if target_size >= 128:
            self.conv_trans_4 = nn.Sequential(
                nn.ConvTranspose2d(feature_size, feature_size // 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_size // 2),
                nn.ReLU(True),
            )
            # Shape [64 x 64]
            feature_size = feature_size // 2

            if target_size == 256:
                self.conv_trans_5 = nn.Sequential(
                    nn.ConvTranspose2d(feature_size, feature_size // 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(feature_size // 2),
                    nn.ReLU(True),
                )
                # Shape [128 x 128]
                feature_size = feature_size // 2

        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(feature_size, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        # Shape [target_size x target_size]

    def forward(self, inputs):
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

class Discriminator(nn.Module):
    '''
    Discriminator model for DCGAN.

    Attributes:
        target_size (int): Target size of input image.
        num_classes (int): Number of classes in dataset.
        feature_size (int): Feature size of the model, defaults to 64. 
    '''
    def __init__(self, target_size, feature_size=64):
        super(Discriminator, self).__init__()
        self.target_size = target_size
        self.feature_size = feature_size

        assert feature_size // 8 > 0, 'Please enter a larger feature size'

        feature_size = feature_size // 2

        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, feature_size, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Dropout(0.2)
        )
        # Shape [target_size / 2 x target_size / 2]

        self.conv_2 = nn.Sequential(
            nn.Conv2d(feature_size, feature_size, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Dropout(0.2),
            nn.BatchNorm2d(feature_size, 0.8)
        )
        # Shape [target_size / 4 x target_size / 4]

        self.conv_3 = nn.Sequential(
            nn.Conv2d(feature_size, feature_size * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Dropout(0.2),
            nn.BatchNorm2d(feature_size * 2, 0.8)
        )
        # Shape [target_size / 8 x target_size / 8]

        feature_size = feature_size * 2

        if target_size >= 128:
            self.conv_4 = nn.Sequential(
                nn.Conv2d(feature_size, feature_size * 2, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace = True),
                nn.Dropout(0.2),
                nn.BatchNorm2d(feature_size * 2, 0.8)
            )

            feature_size = feature_size * 2

            if target_size == 256:
                self.conv_5 = nn.Sequential(
                    nn.Conv2d(feature_size, feature_size * 2, 4, 2, 1),
                    nn.LeakyReLU(0.2, inplace = True),
                    nn.Dropout(0.2),
                    nn.BatchNorm2d(feature_size * 2, 0.8)
                )
                feature_size = feature_size * 2

        self.conv_final = nn.Sequential(
            nn.Conv2d(feature_size, feature_size * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Dropout(0.2),
            nn.BatchNorm2d(feature_size * 2, 0.8)
        )
        feature_size = feature_size * 2

        # Discriminator Layer
        self.discrim = nn.Sequential(
            nn.Linear(feature_size * 4 ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = self.conv_3(x)
        
        if self.target_size >= 128:
            x = self.conv_4(x)
            if self.target_size == 256:
                x = self.conv_5(x)
        
        x = self.conv_final(x)
        input_ = x.view(x.size(0), -1)
        labels = self.discrim(input_)
        return labels