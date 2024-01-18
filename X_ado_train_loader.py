import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def crop(img, h, w):
    img_np = np.array(img)  # Convert PIL Image to numpy array
    ih, iw, _ = img_np.shape
    x = np.random.randint(0, ih - h + 1)
    y = np.random.randint(0, iw - w + 1)
    img_cropped = img_np[x:x+h, y:y+w, :]
    return Image.fromarray(img_cropped)  # Convert back to PIL Image

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor()  # Convert images to tensor after all other transformations
])

class X4K1000FPSDataset(Dataset):
    def __init__(self, root_dir, transform=transform, crop_size=None):
        self.root_dir = root_dir
        self.transform = transform
        self.crop_size = crop_size
        self.image_groups = []

        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                images = sorted(os.listdir(subdir_path))
                for i in range(0, len(images) - 2):
                    self.image_groups.append([os.path.join(subdir_path, images[i]),
                                              os.path.join(subdir_path, images[i+1]),
                                              os.path.join(subdir_path, images[i+2])])

    def __len__(self):
        return len(self.image_groups)

    def __getitem__(self, idx):
        image_paths = self.image_groups[idx]
        images = [Image.open(img_path).convert('RGB') for img_path in image_paths]

        if self.crop_size is not None:
            h, w = self.crop_size
            images = [crop(image, h, w) for image in images]

        images = [self.transform(image) for image in images]
        image_stack = torch.stack(images, dim=0)
        return image_stack

# Create the dataset
dataset = X4K1000FPSDataset(root_dir='/home/jyzhao/Code/Datasets/X4K1000FPS', transform=transform, crop_size=(224, 224))

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
