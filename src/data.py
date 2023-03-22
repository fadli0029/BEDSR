import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImgToTensor(object):
    """
    Converts a PIL.Image to tensor.
    This is to avoid using pytorch
    to_tensor function or ToTensor 
    transform as they automatically
    normalize the input - not what we
    want.
    """
    def __call__(self, pil_image):
        img_np = np.array(pil_image)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
        return img_tensor

class Unsplash(Dataset):
    def __init__(
            self, 
            path, 
            scale, 
            patch_size, 
            mean=np.array([0.0, 0.0, 0.0]), 
            std=np.array([1.0, 1.0, 1.0]), 
            train=True
        ):
        self.path = path
        self.scale = scale
        self.patch_size = patch_size

        if train:
            self.path += 'train_set/'
        else:
            self.path += 'test_set/'
        
        lr = []
        for file in os.listdir(self.path + 'lr_x' + str(self.scale) + '/'):
            if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                continue
            lr.append(file)
        lr = sorted(lr, key=lambda x: int(x.split(".")[0]))

        hr = []
        for file in os.listdir(self.path + 'hr/'):
            if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                continue
            hr.append(file)
        hr = sorted(hr, key=lambda x: int(x.split(".")[0]))

        self.lr = lr
        self.hr = hr
        
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            ImgToTensor()
        ])

    def __len__(self):
        return len(self.hr)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        hr_path = os.path.join(
            self.path + 'hr/', self.hr[idx]
        )
        lr_path = os.path.join(
            self.path + 'lr_x' + str(self.scale) + '/', self.lr[idx]
        )

        hr_img = Image.open(hr_path).convert("RGB")
        lr_img = Image.open(lr_path).convert("RGB")
        
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
