import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
from tqdm.auto import tqdm

from ..utils import authorize_wandb, log_wandb, save_output
from .generator import Generator
from .discriminator import Discriminator


class ACGAN:
    """Auxillary Classifier Generative Adveraisal Network.

    Arguments:
        target_size (int): Target size for the image to be generated.
        num_channels (int): Number of channels in the dataset image.
        num_classes (int): Number of classes in the dataset.
        latent_size (int): Size of the noise. Default: 100.
        generator_feature_size (int): Number of features for the Generator.
        discriminator_feature_size (int): Number of features for the Discriminator.
        g_lr (float): Learning rate for the Generator.
        g_betas (tuple): Co-efficients for the Generator.
        d_lr (float): Learning rate for the Discriminator.
        d_betas (float): Co-efficients for the Discriminator.
    """

    def __init__(
        self,
        target_size,
        num_channels,
        num_classes,
        latent_size=100,
        generator_feature_size=64,
        discriminator_feature_size=64,
        g_lr=0.0002,
        g_betas=(0.5, 0.999),
        d_lr=0.0002,
        d_betas=(0.5, 0.999),
    ):
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.generator = Generator(
            target_size, num_channels, num_classes, latent_size, generator_feature_size
        )
        self.discriminator = Discriminator(
            target_size, num_channels, num_classes, discriminator_feature_size
        )

        self.g_lr = g_lr
        self.g_betas = g_betas
        self.d_lr = d_lr
        self.d_betas = d_betas

        self.g_optim = optim.Adam(
            self.generator.parameters(), lr=self.g_lr, betas=self.g_betas
        )
        self.d_optim = optim.Adam(
            self.discriminator.parameters(), lr=self.d_lr, betas=self.d_betas
        )

        self.device = torch.device("cpu")

    def set_device(self, device):
        """Changes the device on which the models reside.

        Arguments:
            device (torch.device): Device to which the models should switch.

        """
        self.device = device
        self.generator.to(device)
        self.discriminator.to(device)

    def generate(self, labels, inputs=None, output_type="tensor"):
        """Generate images for given labels and inputs.

        Arguments:
            labels (torch.Tensor): A tensor of labels for which \
                the model should generate.
            inputs (None or torch.Tensor): Either give a predefined \
                set of inputs or generate randomly.
            output_type (str): Whether to return a tensor \
                of outputs or a reshaped grid.

        Returns:
            torch.Tensor: Depending on output_type, either the raw output tensors \
                or a tensor grid will be returned.
        """
        if inputs is None:
            inputs = torch.randn(size=(labels.size(0), self.latent_size)).to(
                self.device
            )

        self.generator.eval()
        with torch.no_grad():
            outputs = self.generator(inputs, labels)
        self.generator.train()

        if output_type == "tensor":
            return outputs
        if output_type == "image":
            return torchvision.utils.make_grid(outputs.cpu(), normalize=True)

        raise Exception("Invalid return type specified")

    def save_checkpoint(self, epoch, models_path):
        """Creates a checkpoint of the models and optimizers.

        Arguments:
            epoch (int): Current epoch.
            models_path (str): Path to save current state.
        """
        torch.save(
            {
                "epoch": epoch,
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "g_optim": self.g_optim.state_dict(),
                "d_optim": self.d_optim.state_dict(),
            },
            models_path,
        )

    def load_checkpoint(self, models_path):
        """Load a previously saved checkpoint.

        Arguments:
            models_path (str): Path to load the previous state.

        Returns:
            int: Last processed epoch.
        """
        state = torch.load(models_path, map_location=self.device)

        self.generator.load_state_dict(state["generator"])
        self.discriminator.load_state_dict(state["discriminator"])
        self.g_optim.load_state_dict(state["g_optim"])
        self.d_optim.load_state_dict(state["d_optim"])

        return state["epoch"] + 1

    def train(
        self,
        epochs,
        dataloader,
        epoch_start=0,
        output_batch=64,
        output_epochs=1,
        output_path="./outputs",
        project=None,
        name=None,
        config={},
        models_path=None,
    ):
        """Training loop for ACGAN.

        Arguments:
            epochs (str): Number of epochs for training.
            dataloader (torch.utils.data.DataLoader): \
                PyTorch DataLoader containing the dataset.
            epoch_start (int): The epoch from which training should start.
            output_batch (int): The batch size for the outputs.
            output_epochs (int): The frequency for which outputs \
                will be generated (per epoch).
            output_path (str): The location at which the outputs will be saved. \
                If output_path is wandb, then Weights and Biases will be configured.
            project (str): Project name (Weights and Biases only).
            name (str): Experiment name (Weights and Biases only).
            config (dict): Dictionary containing the configuration settings.
            models_path (str): Path at which (if provided) \
                the checkpoints will be saved.
        """

        if output_path == "wandb":
            if project is None:
                raise Exception("No project name specified")
            authorize_wandb(project, name, config)

        adversarial_loss = nn.BCELoss().to(self.device)
        auxillary_loss = nn.CrossEntropyLoss().to(self.device)

        # Fixed input noise
        fixed_noise = torch.randn(size=(self.num_classes, self.latent_size)).to(
            self.device
        )
        fixed_labels = torch.tensor([i for i in range(self.num_classes)]).to(
            self.device
        )

        # Set tdqm for epoch progress
        pbar = tqdm()

        epoch_end = epochs + epoch_start
        for epoch in range(epoch_start, epoch_end):
            print(f"Epoch: {epoch + 1} / {epoch_end}")
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

                # Generate fake images
                input_noise = torch.randn(
                    size=(current_batch_size, self.latent_size)
                ).to(self.device)
                fake_labels = torch.randint(
                    self.num_classes, size=(current_batch_size,)
                ).to(self.device)

                fake_images = self.generator(input_noise, fake_labels)

                # Calculate Generator loss
                (
                    discriminator_fake_validity,
                    discriminator_fake_labels,
                ) = self.discriminator(fake_images)

                generator_total_loss = (
                    adversarial_loss(discriminator_fake_validity, real_validity)
                    + auxillary_loss(discriminator_fake_labels, fake_labels)
                ) / 2
                generator_total_loss.backward()
                self.g_optim.step()
                generator_total_losses.append(generator_total_loss)

                # Training Discriminator
                self.discriminator.zero_grad()

                # Loss for real images
                (
                    discriminator_real_validity,
                    discriminator_real_labels,
                ) = self.discriminator(real_images)
                discriminator_real_loss = (
                    adversarial_loss(discriminator_real_validity, real_validity)
                    + auxillary_loss(discriminator_real_labels, real_labels)
                ) / 2

                # Loss for fake images
                (
                    discriminator_fake_validity,
                    discriminator_fake_labels,
                ) = self.discriminator(fake_images.detach())
                discriminator_fake_loss = (
                    adversarial_loss(discriminator_fake_validity, fake_validity)
                    + auxillary_loss(discriminator_fake_labels, fake_labels)
                ) / 2

                # Total loss
                discriminator_total_loss = (
                    discriminator_real_loss + discriminator_fake_loss
                )
                discriminator_total_loss.backward()
                self.d_optim.step()
                discriminator_total_losses.append(discriminator_total_loss)

                # Calculate Discriminator Accuracy
                predictions = np.concatenate(
                    [
                        discriminator_real_labels.data.cpu().numpy(),
                        discriminator_fake_labels.data.cpu().numpy(),
                    ],
                    axis=0,
                )
                true_values = np.concatenate(
                    [real_labels.cpu().numpy(), fake_labels.cpu().numpy()], axis=0
                )
                discriminator_accuracy = np.mean(
                    np.argmax(predictions, axis=1) == true_values
                )
                accuracy_history.append(discriminator_accuracy)

                # Update tqdm
                pbar.update()

            d_total_loss = torch.mean(torch.FloatTensor(discriminator_total_losses))
            accuracy = np.mean(accuracy_history)
            g_total_loss = torch.mean(torch.FloatTensor(generator_total_losses))
            print(
                "Discriminator Total Loss: {:.3f}, Discriminator Accuracy: {:.3f}, \
                    Generator Total Loss: {:.3f}".format(
                    d_total_loss, accuracy, g_total_loss
                )
            )

            if output_path == "wandb":
                log_wandb(
                    {
                        "Discriminator Total Loss": d_total_loss,
                        "Discriminator Accuracy": accuracy,
                        "Generator Total Loss": g_total_loss,
                    },
                    epoch + 1,
                )

            if (epoch + 1) % output_epochs == 0:
                save_output(
                    epoch + 1, output_path, fixed_noise, self.generator, fixed_labels
                )
                if models_path:
                    self.save_checkpoint(epoch, models_path)

            pbar.refresh()
