import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import wandb
import os


def authorize_wandb(project, name, config):
    """Authorizes Weights and Biases for the project

    :param project: Name of the project.
    :type project: str
    :param name: Name for the experiment.
    :type name: str
    :param config: Configuration for the run.
    :type config: dict
    """
    wandb.init(project=project, name=name, config=config)


def log_wandb(logs, step):
    """Helper function to save logs to a particular step.

    :param logs: A Python dictionary of the parameters and their values.
    :type logs: dict
    :param step: The current step.
    :type step: int
    """
    wandb.log(logs, step)


def save_output(epoch, output_path, fixed_noise, generator, fixed_labels=None):
    """Save the output of the generator into a specified folder or on Weights and Biases.

    :param epoch: Current epoch.
    :type epoch: int
    :param output_path: Directory to which the image would be saved. \
        If output_path is set to wandb, it will save to your wandb project.
    :type outpt_path: str
    :param fixed_noise: The fixed noise created before training.
    :type fixed_noise: torch.Tensor
    :param generator: The generator model.
    :type generator: Generator
    :param fixed_labels: Labels for conditional generation.
    :type fixed_labels: torch.Tensor
    """
    plt.clf()

    generator.eval()
    with torch.no_grad():

        if fixed_labels is not None:
            test_images = generator(fixed_noise, fixed_labels)
        else:
            test_images = generator(fixed_noise)
    generator.train()

    grid = torchvision.utils.make_grid(test_images.cpu(), normalize=True)

    if output_path == "wandb":
        wandb.log(
            {"output": wandb.Image(grid, caption=f"Output for epoch: {epoch}")},
            step=epoch,
        )
    else:
        image = transforms.ToPILImage()(grid)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        image.save(f"./{output_path}/epoch_{epoch}.jpeg")
