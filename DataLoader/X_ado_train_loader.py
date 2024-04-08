import os
import numpy as np
from PIL import Image
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2 

timestep = 0.5

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
        img0 = crop(images[0],h,w)
        gt = crop(images[1],h,w)
        img1 = crop(images[2],h,w)
        

        '''
        images = [self.transform(image) for image in images]
        image_stack = torch.stack(images, dim=0)
        '''
        # img0, gt, img1 = crop(img0, gt, img1, 224, 224)
        if random.uniform(0, 1) < 0.5:
            img0 = img0[:, :, ::-1]
            img1 = img1[:, :, ::-1]
            gt = gt[:, :, ::-1]
        if random.uniform(0, 1) < 0.5:
            img0 = img0[::-1]
            img1 = img1[::-1]
            gt = gt[::-1]
        if random.uniform(0, 1) < 0.5:
            img0 = img0[:, ::-1]
            img1 = img1[:, ::-1]
            gt = gt[:, ::-1]
        if random.uniform(0, 1) < 0.5:
            tmp = img1
            img1 = img0
            img0 = tmp
            timestep = 1 - timestep
            # random rotation
        p = random.uniform(0, 1)
        if p < 0.25:
            img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
            gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
            img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
        elif p < 0.5:
            img0 = cv2.rotate(img0, cv2.ROTATE_180)
            gt = cv2.rotate(gt, cv2.ROTATE_180)
            img1 = cv2.rotate(img1, cv2.ROTATE_180)
        elif p < 0.75:
            img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
            gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        timestep = torch.tensor(timestep).reshape(1, 1, 1)
        return torch.cat((img0, img1, gt), 0), timestep
        # return image_stack

# Create the dataset
dataset = X4K1000FPSDataset(root_dir='/home/jyzhao/Code/Datasets/X4K1000FPS', transform=transform, crop_size=(224, 224))

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
