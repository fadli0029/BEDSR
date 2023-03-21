#############################################################################################
#############################################################################################
# SUMMARY:
#   This script create a dataset from the Unsplash dataset for Image Super-Resolution task.
#
# DESCRIPTION:
#   Given a path `path` to High Resolution (HR) images in the Unsplash dataset, 
#   this script creates K folders containing bilinear downsampled versions of
#   the HR images based on `scale` passed in the argument.
#   
#   For example, if there are 3 downscaling factors: 2, 4, 6, then K=3. The 3 folders
#   of the following names will be created:
#   - lr_x2
#   - lr_x4
#   - lr_x6
#   They will be placed in `path/`
#
# ARGS:
#   - scale: Downsampling scales to be used.
#   - path: Path to the dataset.
#   - hr: Name of folder in `path` containing HR images.
#
# EXAMPLE:
#   python make_data.py --scale 2 4 6 --path 'dataset/train_set/' --hr 'hr'
#
#############################################################################################
#############################################################################################
import os
from PIL import Image

def get_hr_images(path):
    """
    Return a list of hr images paths.
    """
    hr_images = []
    for file in os.listdir(path):
        if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
            continue
        hr_images.append(file)
    hr_images = sorted(hr_images, key=lambda x: int(x.split(".")[0]))
    return hr_images

def get_lr_images(root_path, hr_images_paths, scale, path_to_save):
    """
    Generate lr images and save it to `path_to_save`
    """
    for hr_img_path in hr_images_paths:
        hr_img = Image.open(root_path + hr_img_path).convert("RGB")

        # Resize image to make sure they're all 1200 x 800 (as original)
        lr_img = hr_img.resize((1200, 800))

        # Compute the new size of the downscaled image and
        # downscale the image using bilinear interpolation.
        new_size = (lr_img.width//scale, lr_img.height//scale)
        lr_img = lr_img.resize(new_size, resample=Image.BILINEAR)

        # Save the image to `path_to_save` as PIL image
        out_path = os.path.join(path_to_save, os.path.basename(hr_img_path))
        lr_img.save(out_path)

path = 'dataset/test_set/'
hr_path = path+'hr/'
lr_path = path+'lr_x'

hr_images_paths = get_hr_images(hr_path)
scales = [2, 4, 6]
for scale in scales:
    lr_path += (str(scale) + '/')
    get_lr_images(hr_path, hr_images_paths, scale=scale, path_to_save=lr_path)
    lr_path = path+'lr_x'
