import torch
import torch.nn as nn

class Discriminator(nn.Module):
    '''
    Discriminator model for DCGAN.

    Attributes:
        target_size (int): Target size of input image.
        num_classes (int): Number of classes in dataset.
        feature_size (int): Feature size of the model, defaults to 64. 
    '''
    def __init__(self, target_size, num_channels, feature_size=64):
        super(Discriminator, self).__init__()
        self.target_size = target_size
        self.feature_size = feature_size

        assert target_size in [64, 128, 256], 'Target size can only be one of the following: [64, 128, 256]'
        assert num_channels in [1, 3], 'Number of channels can only be one of the following: [1, 3]'
        assert feature_size // 8 > 0, 'Please enter a larger feature size'

        feature_size = feature_size // 2

        self.conv_1 = nn.Sequential(
            nn.Conv2d(num_channels, feature_size, 4, 2, 1),
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