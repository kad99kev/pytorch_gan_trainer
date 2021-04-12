This is an example code on how to train a ACGAN in the MNIST Dataset.

```python
import torch
import torchvision
import pytorch_gan_trainer as pgt

image_size = 64
batch_size = 64
epochs = 10
num_channels = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download and prepare dataloader
dataloader, classes = pgt.datasets.prepare_dataloader("mnist", image_size, batch_size)

# Create model
num_classes = len(classes)
acgan = pgt.models.ACGAN(image_size, num_channels, num_classes)
# Set device
acgan.set_device(device)
# Train model
acgan.train(epochs, dataloader, models_path="saved_models.pt")

# Generate images from model
labels = torch.tensor([i for i in range(10)]).to(device)
outputs = acgan.generate(labels)

# Load model
new_acgan = pgt.models.ACGAN(image_size, num_channels, num_classes)
total_epochs = new_acgan.load_checkpoint("saved_models.pt") # Returns the number of epochs the saved model trained for
```
