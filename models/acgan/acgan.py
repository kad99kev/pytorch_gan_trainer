import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm

from ..utils import authorize_wandb, log_wandb, save_output
from .generator import Generator
from .discriminator import Discriminator

class ACGAN:
    def __init__(self, target_size, num_channels, num_classes, latent_size=100, 
        generator_feature_size=64, discriminator_feature_size=64,
        g_lr=0.0002, g_betas=(0.5, 0.999), d_lr=0.0002, d_betas=(0.5, 0.999)
    ):
        
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.generator = Generator(target_size, num_channels, num_classes, latent_size, generator_feature_size)
        self.discriminator = Discriminator(target_size, num_channels, num_classes, discriminator_feature_size)

        self.g_lr = g_lr
        self.g_betas = g_betas
        self.d_lr = d_lr
        self.d_betas = d_betas
        
        self.g_optim = optim.Adam(self.generator.parameters(), lr=self.g_lr, betas=self.g_betas)
        self.d_optim = optim.Adam(self.discriminator.parameters(), lr=self.d_lr, betas=self.d_betas)
        
        self.device = torch.device('cpu')
        

    def set_device(self, device):
        self.device = device
        self.generator.to(device)
        self.discriminator.to(device)
        
    def generate(self, labels, inputs=None, output_type='tensor'):
        if inputs is None:
            inputs = torch.randn(size=(labels.size(0), self.latent_size)).to(self.device)
        
        self.generator.eval()
        with torch.no_grad():
            outputs = self.generator(inputs, labels)
        self.generator.train()
            
        if output_type == 'tensor':
            return outputs
        if output_type == 'image':
            return torchvision.utils.make_grid(outputs.cpu(), normalize=True)
        
        raise Exception('Invalid return type specified')
    
    def save_checkpoint(self, epoch, models_path):
        torch.save({
        'epoch': epoch,
        'generator': self.generator.state_dict(),
        'discriminator': self.discriminator.state_dict(),
        'g_optim': self.g_optim.state_dict(),
        'd_optim': self.d_optim.state_dict()
    }, models_path)
        
    def load_checkpoint(self, models_path):
        state = torch.load(models_path)
        
        self.generator.load_state_dict(state['generator'])
        self.discriminator.load_state_dict(state['discriminator'])
        self.g_optim.load_state_dict(state['g_optim'])
        self.d_optim.load_state_dict(state['d_optim'])
        
        return state['epoch']

    
    def train(self, epochs, dataloader, epoch_start=0, output_batch=64, output_epochs=1, output_path='./outputs', project=None, name=None, config={}, models_path=None):

        if output_path == 'wandb':
            if project is None:
                raise Exception('No project name specified')
            authorize_wandb(project, name, config)

        adversarial_loss = nn.BCELoss().to(self.device)
        auxillary_loss = nn.CrossEntropyLoss().to(self.device)

        # Fixed input noise
        fixed_noise = torch.randn(size=(self.num_classes, self.latent_size)).to(self.device)
        fixed_labels = torch.tensor([i for i in range(self.num_classes)]).to(self.device)

        # Set tdqm for epoch progress
        pbar = tqdm()

        for epoch in range(epoch_start, epochs + epoch_start):
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
                self.g_optim.step()
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
                self.d_optim.step()
                discriminator_total_losses.append(discriminator_total_loss)

                # Calculate Discriminator Accuracy
                predictions = np.concatenate([discriminator_real_labels.data.cpu().numpy(), discriminator_fake_labels.data.cpu().numpy()], axis=0)
                true_values = np.concatenate([real_labels.cpu().numpy(), fake_labels.cpu().numpy()], axis=0)
                discriminator_accuracy = np.mean(np.argmax(predictions, axis=1) == true_values)
                accuracy_history.append(discriminator_accuracy)

                # Update tqdm
                pbar.update()

            d_total_loss = torch.mean(torch.FloatTensor(discriminator_total_losses))
            accuracy = np.mean(accuracy_history)
            g_total_loss = torch.mean(torch.FloatTensor(generator_total_losses))
            print('Discriminator Total Loss: {:.3f}, Discriminator Accuracy: {:.3f}, Generator Total Loss: {:.3f}'.format(
                d_total_loss,
                accuracy, 
                g_total_loss
            ))

            if output_path == 'wandb':
                log_wandb({
                    'Discriminator Total Loss': d_total_loss,
                    'Discriminator Accuracy': accuracy,
                    'Generator Total Loss': g_total_loss
                    }, epoch + 1)

            if (epoch + 1) % output_epochs == 0:
                save_output(epoch + 1, output_path, fixed_noise, self.generator, fixed_labels)
                if models_path: self.save_checkpoint(epoch, models_path)

            pbar.refresh()