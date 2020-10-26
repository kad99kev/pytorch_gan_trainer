import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from ..utils import authorize_wandb, log_wandb, save_output
from .generator import Generator
from .discriminator import Discriminator

class DCGAN:
    def __init__(self, target_size, num_channels, latent_size=100, 
        generator_feature_size=64, discriminator_feature_size=64,
        g_lr=0.0002, g_betas=(0.5, 0.999), d_lr=0.0002, d_betas=(0.5, 0.999)
    ):
        
        self.latent_size = latent_size
        self.generator = Generator(target_size, num_channels, latent_size, generator_feature_size)
        self.discriminator = Discriminator(target_size, num_channels, discriminator_feature_size)

        self.g_lr = g_lr
        self.g_betas = g_betas
        self.d_lr = d_lr
        self.d_betas = d_betas

    def set_device(self, device):
        self.device = device
        self.generator.to(device)
        self.discriminator.to(device)
        
    def generate(self, inputs=None, output_type='tensor'):
        if inputs is None:
            inputs = torch.randn(size=(current_batch_size, self.latent_size)).to(self.device)
        self.generator.eval()
        with torch.no_grad():
            outputs = self.generator(inputs)
            
        if output_type == 'tensor':
            return outputs
        if output_type == 'image':
            return torchvision.utils.make_grid(outputs.cpu(), normalize=True)
        
        raise Exception('Invalid return type specified')
    
    def train(self, epochs, dataloader, output_batch=64, output_epochs=1, output_path='./outputs', project=None, id=None, config={}, models_path=None):
        
        if output_path == 'wandb':
            if project is None:
                raise Exception('No project name specified')
            authorize_wandb(project, name, config)

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

            d_total_loss = torch.mean(torch.FloatTensor(discriminator_total_losses))
            g_total_loss = torch.mean(torch.FloatTensor(generator_total_losses))
            print('Discriminator Total Loss: {:.3f}, Generator Total Loss: {:.3f}'.format(
                    d_total_loss,
                    g_total_loss
                ))

            if output_path == 'wandb':
                log_wandb({
                    'Discriminator Total Loss': d_total_loss,
                    'Generator Total Loss': g_total_loss
                    }, epoch + 1)

            if (epoch + 1) % output_epochs == 0:
                save_output(epoch + 1, output_path, fixed_noise, self.generator)
                if models_path: save_checkpoint(epoch, models_path, self.generator, self.discriminator, g_optimizer, d_optimizer)

            pbar.refresh()