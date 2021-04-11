import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm.auto import tqdm

from ..utils import authorize_wandb, log_wandb, save_output
from .generator import Generator
from .discriminator import Discriminator


class DCGAN:
    """Deep Convolutional Generative Adveraisal Network."""

    def __init__(
        self,
        target_size,
        num_channels,
        latent_size=100,
        generator_feature_size=64,
        discriminator_feature_size=64,
        g_lr=0.0002,
        g_betas=(0.5, 0.999),
        d_lr=0.0002,
        d_betas=(0.5, 0.999),
    ):
        """Initialization function for DCGAN.

        :param target_size: Target size for the image to be generated.
        :type target_size: int
        :param num_channels: Number of channels in the dataset image.
        :type num_channels: int
        :param latent_size: Size of the noise. Default: 100.
        :type latent_size: int
        :param generator_feature_size: Number of features for the Generator.
        :type generator_feature_size: int
        :param discriminator_feature_size: Number of features for the Discriminator.
        :type discriminator_feature_size: int
        :param g_lr: Learning rate for the Generator.
        :type g_lr: float
        :param g_betas: Co-efficients for the Generator.
        :type g_betas: tuple
        :param d_lr: Learning rate for the Discriminator.
        :type d_lr: float
        :param d_betas: Co-efficients for the Discriminator.
        :type d_betas: float
        """

        self.latent_size = latent_size
        self.generator = Generator(
            target_size, num_channels, latent_size, generator_feature_size
        )
        self.discriminator = Discriminator(
            target_size, num_channels, discriminator_feature_size
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

        :param device: Device to which the models should switch.
        :type device: torch.device
        """
        self.device = device
        self.generator.to(device)
        self.discriminator.to(device)

    def generate(self, batch_size, inputs=None, output_type="tensor"):
        """Generate images for given labels and inputs.

        :param batch_size: Batch size of outputs.
        :type batch_size: int
        :param inputs: Either give a predefined set of inputs or generate randomly.
        :type inputs: None or torch.Tensor
        :param output_type: Whether to return a tensor of outputs or a reshaped grid.
        :type output_type: str
        :returns: torch.Tensor
        """
        if inputs is None:
            inputs = torch.randn(size=(batch_size, self.latent_size)).to(self.device)

        self.generator.eval()
        with torch.no_grad():
            outputs = self.generator(inputs)
        self.generator.train()

        if output_type == "tensor":
            return outputs
        if output_type == "image":
            return torchvision.utils.make_grid(outputs.cpu(), normalize=True)

        raise Exception("Invalid return type specified")

    def save_checkpoint(self, epoch, models_path):
        """Creates a checkpoint of the models and optimizers.

        :param epoch: Current epoch.
        :type epoch: int
        :param models_path: Path to save current state.
        :type models_path: str
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
        :param models_path: Path to load the previous state.
        :type models_path: str
        :returns: Last processed epoch.
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
        """Training loop for DCGAN.

        :param epochs: Number of epochs for training.
        :type epochs: int
        :param dataloader: PyTorch DataLoader containing the dataset.
        :type dataloader: DataLoader
        :param epoch_start: The epoch from which training should start.
        :type epoch_start: int
        :param output_batch: The batch size for the outputs.
        :type output_batch: int
        :param output_epochs: The frequency for which outputs \
            will be generated (per epoch).
        :type output_epochs: int
        :param output_path: The location at which the outputs will be saved. \
            If output_path is wandb, then Weights and Biases will be configured.
        :type output_path: str
        :param project: Project name (Weights and Biases only).
        :type project: str
        :param name: Experiment name (Weights and Biases only).
        :type name: str
        :param config: Dictionary containing the configuration settings.
        :type config: dict
        :param models_path: Path at which (if provided) the checkpoints will be saved.
        :type models_path: str
        """

        if output_path == "wandb":
            if project is None:
                raise Exception("No project name specified")
            authorize_wandb(project, name, config)

        adversarial_loss = nn.BCELoss().to(self.device)

        # Fixed input noise
        fixed_noise = torch.randn(size=(output_batch, self.latent_size)).to(self.device)

        # Set tdqm for epoch progress
        pbar = tqdm()

        epoch_end = epochs + epoch_start
        for epoch in range(epoch_start, epoch_end):
            print(f"Epoch: {epoch + 1} / {epoch_end}")
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

                # Generate fake images
                input_noise = torch.randn(
                    size=(current_batch_size, self.latent_size)
                ).to(self.device)

                fake_images = self.generator(input_noise)

                # Calculate Generator loss
                discriminator_fake_labels = self.discriminator(fake_images)

                generator_total_loss = adversarial_loss(
                    discriminator_fake_labels, real_labels
                )
                generator_total_loss.backward()
                self.g_optim.step()
                generator_total_losses.append(generator_total_loss)

                # Training Discriminator
                self.discriminator.zero_grad()

                # Loss for real images
                discriminator_real_labels = self.discriminator(real_images)
                discriminator_real_loss = adversarial_loss(
                    discriminator_real_labels, real_labels
                )

                # Loss for fake images
                discriminator_fake_labels = self.discriminator(fake_images.detach())
                discriminator_fake_loss = adversarial_loss(
                    discriminator_fake_labels, fake_labels
                )

                # Total loss
                discriminator_total_loss = (
                    discriminator_real_loss + discriminator_fake_loss
                )
                discriminator_total_loss.backward()
                self.d_optim.step()
                discriminator_total_losses.append(discriminator_total_loss)

                # Update tqdm
                pbar.update()

            d_total_loss = torch.mean(torch.FloatTensor(discriminator_total_losses))
            g_total_loss = torch.mean(torch.FloatTensor(generator_total_losses))
            print(
                "Discriminator Total Loss: {:.3f}, Generator Total Loss: {:.3f}".format(
                    d_total_loss, g_total_loss
                )
            )

            if output_path == "wandb":
                log_wandb(
                    {
                        "Discriminator Total Loss": d_total_loss,
                        "Generator Total Loss": g_total_loss,
                    },
                    epoch + 1,
                )

            if (epoch + 1) % output_epochs == 0:
                save_output(epoch + 1, output_path, fixed_noise, self.generator)
                if models_path:
                    self.save_checkpoint(epoch, models_path)

            pbar.refresh()
