import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class DCGAN:
    def __init__(self, target_size, latent_size=100, 
        generator_feature_size=64, discriminator_feature_size=64,
        g_lr=0.0002, g_betas=(0.5, 0.999), d_lr=0.0002, d_betas=(0.5, 0.999)
    ):
        
        self.latent_size = latent_size
        self.generator = Generator(target_size, latent_size, generator_feature_size)
        self.discriminator = Discriminator(target_size, discriminator_feature_size)

        self.g_lr = g_lr
        self.g_betas = g_betas
        self.d_lr = d_lr
        self.d_betas = d_betas

    def set_device(self, device):
        self.device = device
        self.generator.to(device)
        self.discriminator.to(device)
    
    def train(self, epochs, dataloader, output_batch=64, output_epochs=1, models_path=None):
        
        g_optimizer = optim.Adam(self.generator.parameters(), lr=self.g_lr, betas=self.g_betas)
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.d_lr, betas=self.d_betas)

        if not self.device:
            self.device = torch.device('cpu')

        adversarial_loss = nn.BCELoss().to(self.device)

        # Fixed input noise
        fixed_noise = torch.randn(size=(output_batch, self.latent_size)).to(self.device)

        # Set tdqm for epoch progress
        pbar = tqdm()

        for epoch in range(epochs):
            print(f'Epoch: {epoch + 1} / {epochs}')
            pbar.reset(total=len(dataloader))

            # Setting up losses
            discriminator_total_losses = []
            generator_total_losses = []

            for real_images, _ in dataloader:

                # Current batch size
                current_batch_size = real_images.size()[0]

                # Convert to cuda
                real_images = real_images.to(self.device)

                # For real vs fake
                real_labels = torch.ones(current_batch_size, 1).to(self.device)
                fake_labels = torch.zeros(current_batch_size, 1).to(self.device)

                # Training Generator
                self.generator.zero_grad()

                ## Generate fake images
                input_noise = torch.randn(size=(current_batch_size, self.latent_size)).to(self.device)

                fake_images = self.generator(input_noise)
                print(f'Fake image shape: {fake_images.shape}')

                ## Calculate Generator loss
                discriminator_fake_labels = self.discriminator(fake_images)
                
                generator_total_loss = adversarial_loss(discriminator_fake_labels, real_labels)
                generator_total_loss.backward()
                g_optimizer.step()
                generator_total_losses.append(generator_total_loss)

                # Training Discriminator
                self.discriminator.zero_grad()

                ## Loss for real images
                discriminator_real_labels = self.discriminator(real_images)
                discriminator_real_loss = adversarial_loss(discriminator_real_labels, real_labels)

                ## Loss for fake images
                discriminator_fake_labels = self.discriminator(fake_images.detach())
                discriminator_fake_loss = adversarial_loss(discriminator_fake_labels, fake_labels)

                ## Total loss
                discriminator_total_loss = discriminator_real_loss + discriminator_fake_loss
                discriminator_total_loss.backward()
                d_optimizer.step()
                discriminator_total_losses.append(discriminator_total_loss)

                # Update tqdm
                pbar.update()

            print('Discriminator Total Loss: {:.3f}, Generator Total Loss: {:.3f}'.format(
                    torch.mean(torch.FloatTensor(discriminator_total_losses)),
                    torch.mean(torch.FloatTensor(generator_total_losses))
                ))

            if (epoch + 1) % output_epochs == 0:
                plot_output(fixed_noise, generator)

            pbar.refresh()

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
    Discriminator model for ACGAN.

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