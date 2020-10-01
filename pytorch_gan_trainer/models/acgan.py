import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

class ACGAN:
    def __init__(self, target_size, num_classes, latent_size=100, 
        generator_feature_size=64, discriminator_feature_size=64,
        g_lr=0.0002, g_betas=(0.5, 0.999), d_lr=0.0002, d_betas=(0.5, 0.999)
    ):
        
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.generator = Generator(target_size, num_classes, latent_size, generator_feature_size)
        self.discriminator = Discriminator(target_size, num_classes, discriminator_feature_size)

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

        adversarial_loss = nn.BCELoss().to(self.device)
        auxillary_loss = nn.CrossEntropyLoss().to(self.device)

        # Fixed input noise
        fixed_noise = torch.randn(size=(self.num_classes, self.latent_size)).to(self.device)
        fixed_labels = torch.tensor([i for i in range(self.num_classes)]).to(self.device)

        # Set tdqm for epoch progress
        pbar = tqdm()

        for epoch in range(epochs):
            print(f'Epoch: {epoch + 1} / {epochs}')
            pbar.reset(total=len(dataloader))

            # Setting up losses
            discriminator_total_losses = []
            generator_total_losses = []
            accuracy_history = []

            for real_images, real_labels in dataloader:

                # Current batch size
                current_batch_size = real_images.size()[0]

                # Convert to cuda
                real_images = real_images.to(self.device)
                real_labels = real_labels.to(self.device)

                # For real vs fake
                real_validity = torch.ones(current_batch_size, 1).to(self.device)
                fake_validity = torch.zeros(current_batch_size, 1).to(self.device)

                # Training Generator
                self.generator.zero_grad()

                ## Generate fake images
                input_noise = torch.randn(size=(current_batch_size, self.latent_size)).to(self.device)
                fake_labels = torch.randint(self.num_classes, size=(current_batch_size, )).to(self.device)

                fake_images = self.generator(input_noise, fake_labels)

                ## Calculate Generator loss
                discriminator_fake_validity, discriminator_fake_labels = self.discriminator(fake_images)
                
                generator_total_loss = (adversarial_loss(discriminator_fake_validity, real_validity) + auxillary_loss(discriminator_fake_labels, fake_labels)) / 2
                generator_total_loss.backward()
                g_optimizer.step()
                generator_total_losses.append(generator_total_loss)

                # Training Discriminator
                self.discriminator.zero_grad()

                ## Loss for real images
                discriminator_real_validity, discriminator_real_labels = self.discriminator(real_images)
                discriminator_real_loss = (adversarial_loss(discriminator_real_validity, real_validity) + auxillary_loss(discriminator_real_labels, real_labels)) / 2

                ## Loss for fake images
                discriminator_fake_validity, discriminator_fake_labels = self.discriminator(fake_images.detach())
                discriminator_fake_loss = (adversarial_loss(discriminator_fake_validity, fake_validity) + auxillary_loss(discriminator_fake_labels, fake_labels)) / 2

                ## Total loss
                discriminator_total_loss = discriminator_real_loss + discriminator_fake_loss
                discriminator_total_loss.backward()
                d_optimizer.step()
                discriminator_total_losses.append(discriminator_total_loss)

                # Calculate Discriminator Accuracy
                predictions = np.concatenate([discriminator_real_labels.data.cpu().numpy(), discriminator_fake_labels.data.cpu().numpy()], axis=0)
                true_values = np.concatenate([real_labels.cpu().numpy(), fake_labels.cpu().numpy()], axis=0)
                discriminator_accuracy = np.mean(np.argmax(predictions, axis=1) == true_values)
                accuracy_history.append(discriminator_accuracy)

                # Update tqdm
                pbar.update()

            print('Discriminator Total Loss: {:.3f}, Discriminator Accuracy: {:.3f}, Generator Total Loss: {:.3f}'.format(
                    torch.mean(torch.FloatTensor(discriminator_total_losses)),
                    np.mean(accuracy_history), 
                    torch.mean(torch.FloatTensor(generator_total_losses))
                ))

            if (epoch + 1) % output_epochs == 0:
                plot_output(fixed_noise, fixed_labels, generator)

            pbar.refresh()

class Generator(nn.Module):
    '''
    Generator model for ACGAN.

    Attributes:
        target_size (int): Target size of output image.
        num_classes (int): Number of classes in dataset.
        latent_size (int): Size of input noise, defaults to 100.
        feature_size (int): Feature size of the model, defaults to 64. 
    '''

    def __init__(self, target_size, num_classes, latent_size, feature_size):
        super(Generator, self).__init__()

        assert target_size in [64, 128, 256], 'Target size can only be one of the following: [64, 128, 256]'

        self.target_size = target_size
        self.feature_size = feature_size

        # For labels
        self.label_embedding = nn.Embedding(num_classes, latent_size)
        self.linear = nn.Sequential(nn.Linear(latent_size, (feature_size * 2) * 4 ** 2))

        self.norm_layer = nn.Sequential(
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

    def forward(self, inputs, labels):
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


class Discriminator(nn.Module):
    '''
    Discriminator model for ACGAN.

    Attributes:
        target_size (int): Target size of input image.
        num_classes (int): Number of classes in dataset.
        feature_size (int): Feature size of the model, defaults to 64. 
    '''
    def __init__(self, target_size, num_classes, feature_size=64):
        super(Discriminator, self).__init__()
        self.target_size = target_size
        self.feature_size = feature_size
        self.num_classes = num_classes

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

        # Auxillary Layer
        self.aux = nn.Sequential(
            nn.Linear(feature_size * 4 ** 2, num_classes),
            nn.Softmax(dim=1))

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
        labels = self.aux(input_)
        valid = self.discrim(input_)
        return valid, labels