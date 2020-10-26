import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os

def authorize_wandb(project, name, config):
    wandb.init(project=project, name=name, config=config)

def log_wandb(logs, step):
    wandb.log(logs, step)

def save_output(epoch, output_path, fixed_noise, generator, fixed_labels=None):
    plt.clf()
    
    generator.eval()
    with torch.no_grad():
    
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
        
def save_checkpoint(epoch, model_path, generator, discriminator, g_optim, d_optim):
    torch.save({
        'epoch': epoch,
        'generator': generator,
        'discriminator': discriminator,
        'g_optim': g_optim,
        'd_optim': d_optim
    }, model_path)