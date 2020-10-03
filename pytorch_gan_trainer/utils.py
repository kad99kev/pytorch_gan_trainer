import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os

def authorize_wandb(project, id, config):
    wandb.init(project=project, id=id, config=config)

def plot_output(epoch, output_path, fixed_noise, generator, fixed_labels=None):
    plt.clf()
    with torch.no_grad():
    
        generator.eval()
        if fixed_labels is not None:
            test_images = generator(fixed_noise, fixed_labels)
        else:
            test_images = generator(fixed_noise)
        generator.train()

    grid = torchvision.utils.make_grid(test_images.cpu(), normalize=True)
    
    if output_path == 'wandb':
        wandb.log({'output': wandb.Image(grid, caption=f'Output for epoch: {epoch}')}, step=epoch)
    else:
        image = transforms.ToPILImage()(grid)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        image.save(f'./{output_path}/epoch_{epoch}.jpeg')