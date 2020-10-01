import numpy as np
import matplotlib.pyplot as plt

def plot_output(fixed_noise, generator, fixed_labels=None):
    plt.clf()
    with torch.no_grad():
    
        generator.eval()
        if fixed_labels:
            test_images = generator(fixed_noise, fixed_labels)
        else:
            test_images = generator(fixed_noise)

        generator.train()

    grid = torchvision.utils.make_grid(test_images.cpu(), normalize=True)
    _show_grid(grid)

def _show_grid(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()