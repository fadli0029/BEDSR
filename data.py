import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class Unsplash(Dataset):
    def __init__(self, path, scale, patch_size, mean, std):
        self.path = path
        self.patch_size = patch_size
        self.scale = scale
        
        labels = []
        for file in os.listdir(self.path):
            if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                continue
            labels.append(file)
        labels = sorted(labels, key=lambda x: int(x.split(".")[0]))

        self.images = labels
        self.labels = labels
        
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Note: The Unsplash dataset is differet. Its downsampled
        # low res images has the same dimension as the high res
        # images. This is possible since the dataset author may have 
        # used a downsampling method that preserves the image size while 
        # reducing the resolution (ex: nearest-neighbor interpolation).
        
        # As a consequence:
        # We'll downsampled the hr image.
        
        if torch.is_tensor(idx):
            idx = idx.to_list()

        label_path = os.path.join(
            self.path, self.labels[idx]
        )

        hr_img = Image.open(label_path).convert("RGB")
        lr_img = hr_img
        
        # Resize image to make sure they're all 1200 x 800 (as original)
        lr_img = lr_img.resize((1200, 800))
        hr_img = hr_img.resize((1200, 800))
        
        # Compute the new size of the downscaled image and
        # downscale the image using bilinear interpolation.
        new_size = (lr_img.width//self.scale, lr_img.height//self.scale)
        lr_img = lr_img.resize(new_size, resample=Image.BILINEAR)
        
        # Get random patch location of size self.patch_size.
        lr_patch_x = np.random.randint(0, lr_img.size[0] - self.patch_size)
        lr_patch_y = np.random.randint(0, lr_img.size[1] - self.patch_size)
        hr_patch_x = lr_patch_x * self.scale
        hr_patch_y = lr_patch_y * self.scale
        
        
        # Get the patch from both lr and hr image.
        lr_patch = lr_img.crop(
            (lr_patch_x, 
             lr_patch_y, 
             lr_patch_x + self.patch_size, 
             lr_patch_y + self.patch_size)
        )
        hr_patch = hr_img.crop(
            (hr_patch_x, 
             hr_patch_y, 
             hr_patch_x + (self.patch_size*self.scale),
             hr_patch_y + (self.patch_size*self.scale))
        )
        
        hr_tensor = self.transform(hr_patch)
        lr_tensor = self.transform(lr_patch)
        return lr_tensor, hr_tensor
