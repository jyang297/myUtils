import os
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class X4K1000FPSDataset(Dataset):
    def __init__(self, root_dir, transform=None, crop=None, ):
        self.root_dir = root_dir
        self.crop = crop
        self.transform = transform
        self.image_groups = []
        self.transformer = transformImages()

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
        images = [self.transform(image) for image in images]


        if self.transform:
            for image in images:
                if self.crop:
                    image = self.transformImages()
            

        image_stack = torch.stack(images, dim=0)
        return image_stack

# Define transformations, if necessary
transform = transforms.Compose([
    transforms.ToTensor(),
])
# Create the dataset
dataset = X4K1000FPSDataset(root_dir='/home/jyzhao/Code/Datasets/X4K1000FPS', transform=transform)

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Now you can use this dataloader in your training loop

class transformImages():
    def crop(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        # img1 = img1[x:x+h, y:y+w, :]
        # gt = gt[x:x+h, y:y+w, :]
        return img0
    
    
    def RGB2Gray(image, channel=3):
    ## input : T, H, W, C

        # rgb --> Y (gray)
        r, g, b = image[0], image[1], image[2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        # Add channel dimension
        gray = gray.unsqueeze(0)
        return gray
    def normalization(image, range=None):
        if range:
            # normalization [-1,1]
            imgIn = (imgIn / 255.0 - 0.5) * 2
        else:
            # normalization [0,1]
            imgIn = (imgIn / 255.0)
    
        return imgIn

def NGNtripletSetup(_):
    return None


 
