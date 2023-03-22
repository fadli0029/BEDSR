import torch
import numpy as np
from PIL import Image, ImageDraw

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

def output_to_image(output):
    output = output[0, :, :, :]
    output = np.array(output)
    output = np.transpose(output, (1, 2, 0))
    image = Image.fromarray(output.astype(np.uint8), mode='RGB')
    return image

def image_to_tensor(img):
    img_np = np.array(img)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
    return img_tensor

def calc_psnr(lr_img, hr_img, max_val=255.0):
    """
    Calculates the peak signal-to-noise ratio (PSNR) 
    between a low-resolution (LR) image and a high-resolution (HR) image.

    Parameters:
    - lr_img: torch tensor representing the LR image.
    - hr_img: torch tensor representing the HR image.

    Returns:
    - PSNR value as a float.
    """
    from torchmetrics import PeakSignalNoiseRatio

    # Check that the images have the same shape
    if lr_img.shape != hr_img.shape:
        raise ValueError("LR and HR images must have the same size.")
    psnr = PeakSignalNoiseRatio(data_range=max_val).to(0)
    return psnr(lr_img, hr_img)

def img_to_patch(imgs, scale=2, patch_size=48):
    """
    Given (test) image pairs `imgs` of
    type PIL.Image, take a random
    patch from `img`, return this patch as
    input to the network.
    """
    lr_img, hr_img = imgs
    
    # Get random patch location of size patch_size.
    lr_patch_x = np.random.randint(0, lr_img.size[0] - patch_size)
    lr_patch_y = np.random.randint(0, lr_img.size[1] - patch_size)
    hr_patch_x = lr_patch_x * scale
    hr_patch_y = lr_patch_y * scale

    lr_a, lr_b, lr_c, lr_d = \
        lr_patch_x, lr_patch_y, \
        lr_patch_x + patch_size, \
        lr_patch_y + patch_size

    hr_a, hr_b, hr_c, hr_d = \
        hr_patch_x, hr_patch_y, \
        hr_patch_x + (patch_size*scale), \
        hr_patch_y + (patch_size*scale)
        
    # Get the patch from both lr and hr image.
    lr_patch = lr_img.crop(
        (lr_a, lr_b, lr_c, lr_d)
    )
    hr_patch = hr_img.crop(
        (hr_a, hr_b, hr_c, hr_d)
    )

    patch_coord = (
        (lr_a, lr_b, lr_c, lr_d),
        (hr_a, hr_b, hr_c, hr_d)
    )
    return lr_patch, hr_patch, patch_coord

def annotated_image(imgs, patch_coord, width=(3, 6)):
    """
    Given a full `imgs` pair return `annotated_img`
    that contains a bounding box at `patch_coord`.
    It serves to show the location of
    choosen patch on the image.
    """
    lr_img, hr_img = imgs
    annot_lr = ImageDraw.Draw(lr_img)
    annot_hr = ImageDraw.Draw(hr_img)
    
    lrs, hrs = patch_coord
    
    annot_lr.rectangle(
        [(lrs[0], lrs[1]), (lrs[2], lrs[3])],
        outline='red',
        width=width[0]
    )
    annot_hr.rectangle(
        [(hrs[0], hrs[1]), (hrs[2], hrs[3])],
        outline='red',
        width=width[1]
    )
    return lr_img, hr_img
