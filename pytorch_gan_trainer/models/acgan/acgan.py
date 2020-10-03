import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from pytorch_gan_trainer.utils import authorize_wandb, plot_output
from tqdm.auto import tqdm
from .models import Generator, Discriminator

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
    
    
    def train(self, epochs, dataloader, output_batch=64, output_epochs=1, output_path='./outputs', project=None, id=None, config={}, models_path=None):

        if output_path == 'wandb':
            if project is None or id is None:
                raise Exception('No project name or id specified')
            authorize_wandb(project, id, config=config)
        
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
                plot_output(epoch + 1, output_path, fixed_noise, self.generator, fixed_labels)

            pbar.refresh()