import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, ToPILImage, Resize
from torch.utils.data import Dataset, DataLoader

# Define the U-Net model architecture
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Add more encoder layers as needed
        )

        # Decoder
        self.decoder = nn.Sequential(
            # Add decoder layers as needed
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Implement the forward pass of your model
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, fluorescence_folder, mask_folder):
        self.fluorescence_files = sorted(glob.glob(os.path.join(fluorescence_folder, '*.tif')))
        self.mask_files = sorted(glob.glob(os.path.join(mask_folder, '*.tif')))
        self.transform = ToTensor()
        self.resize = Resize((256, 256))
        self.to_pil = ToPILImage()

    def __len__(self):
        return len(self.fluorescence_files)

    def __getitem__(self, idx):
        fluorescence_image = plt.imread(self.fluorescence_files[idx])
        mask_image = plt.imread(self.mask_files[idx])

        fluorescence_image = self.to_pil(fluorescence_image)
        mask_image = self.to_pil(mask_image).convert('L')  # Convert mask to grayscale

        fluorescence_image = self.transform(self.resize(fluorescence_image))
        mask_image = self.transform(self.resize(mask_image))

        return fluorescence_image, mask_image

# Set the paths to your fluorescence and mask folders
fluorescence_folder = 'train/fluorescence'
mask_folder = 'train/masks'

# Create the dataset
dataset = CustomDataset(fluorescence_folder, mask_folder)

# Create a data loader
batch_size = 4  # Adjust as needed
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create an instance of the U-Net model
model = UNet()

# Define your loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10  # Adjust as needed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    for fluorescence_batch, mask_batch in data_loader:
        fluorescence_batch = fluorescence_batch.to(device)
        mask_batch = mask_batch.to(device)

        # Forward pass
        outputs = model(fluorescence_batch)

        # Resize the output to match the size of the ground truth mask
        outputs = torch.nn.functional.interpolate(outputs, size=(256, 256), mode='bilinear', align_corners=False)

        # Compute the loss
        loss = loss_fn(outputs, mask_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Set the path to your test image
test_image_path = 'test/fluorescence/test_1.tif'

# Load the test image
test_image = plt.imread(test_image_path)

# Preprocess the test image
test_image = ToPILImage()(test_image)
test_image = Resize((256, 256))(test_image)
test_image = ToTensor()(test_image)
test_image = test_image.unsqueeze(0).to(device)

# Make predictions on the test image
with torch.no_grad():
    output_mask = model(test_image)
    output_mask = torch.nn.functional.interpolate(output_mask, size=(256, 256), mode='bilinear', align_corners=False)
    output_mask = output_mask.cpu().squeeze().numpy()

# Visualize the test image and the predicted mask
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(test_image.squeeze().cpu().numpy(), cmap='gray')
axes[0].set_title('Test Image')
axes[1].imshow(output_mask, cmap='gray')
axes[1].set_title('Predicted Mask')
plt.show()
