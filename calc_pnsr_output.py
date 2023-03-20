from __future__ import print_function

from PIL import Image
from os import listdir
from torchvision import transforms
import argparse
import torch
import numpy as np
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from dataset import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input_image', type=str, default='dataset/test_set/lr_x2/1188.jpg', help='input image to use')
parser.add_argument('--scale', type=int, default=2, help='downsampling scale of input image')
parser.add_argument('--input_LR_path', type=str, default='dataset/test_set/lr_x', help='input path to use')
parser.add_argument('--input_HR_path', type=str, default='dataset/test_set/hr/', help='input path to use')
parser.add_argument('--model', type=str, default='checkpoints/model_epoch_150.pth', help='model file to use')
parser.add_argument('--output_path', default='results/', type=str, help='where to save the output image')
parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')
opt = parser.parse_args()

print(opt)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

def calc_psnr(img1, img2):
    """
    img1: Generated image (float).
    img2: Ground truth (hr) image (float).
    """
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

def annotated_image(img, patch_coord):
    # TODO
    """
    Given an `img` return `annotated_img`
    that contains a bounding box at `patch_coord`.
    It serves to show the location of
    choosen patch on the image.
    """
    pass

def img_to_patches(img, patch_size=48):
    # TODO
    """
    Given a (test) image `img`, split the
    images into patches of (`patch_size`, patch_size).
    The, return these patches as input to network.
    """
    pass

def stitch_patches(patches):
    # TODO
    """
    Given patches, stich them into
    a full image.
    """
    pass

loader = transforms.Compose([
    transforms.ToTensor()])

scale = opt.scale
path = opt.input_LR_path + str(scale) + '/'
path_HR = opt.input_HR_path

image_nums = len([lists for lists in listdir(path) if is_image_file('{}/{}'.format(path, lists))])
print("Number of images:", image_nums)

psnr_avg = 0
for i in listdir(path):
    if is_image_file(i):
        with torch.no_grad():
            img_name = i.split('.')
            img_num = img_name[0]

            img_original = Image.open('{}{}'.format(path_HR, i)).convert('RGB')
            img_original = img_original.resize((1200, 800))

            img_LR = Image.open('{}{}'.format(path, i)).convert('RGB')

            img_to_tensor = ToTensor()
            input = img_to_tensor(img_LR)
            input = Variable(torch.unsqueeze(input, dim=0).float(), requires_grad=False)

            model = torch.load(opt.model, map_location='cuda:0')
            if opt.cuda:
                model = model.cuda()
                input = input.cuda()

            out = model(input)
            out = out.cpu()

            im_h = out.data[0].numpy().astype(np.float32)
            im_h = im_h * 255.
            im_h = np.clip(im_h, 0., 255.)
            im_h = im_h.transpose(1, 2, 0)
            im_h_pil = Image.fromarray(im_h.astype(np.uint8))
            im_h_pil_ybr = im_h_pil.convert('YCbCr')
            im_h_pil_y, _, _ = im_h_pil_ybr.split()

            psnr_val = calc_psnr(loader(im_h_pil_y), loader(img_original_y))
            psnr_avg += psnr_val
            print("Test image: {}   ===> PSNR: {}".format(img_num, psnr_val))

            im_h_pil.save('{}predicted_images/{}.png'.format(opt.output_path, img_num))
            img_original.save('{}original_images/{}.png'.format(opt.output_path, img_num))
psnr_avg = psnr_avg / image_nums
print('AVERAGE PSNR:', psnr_avg)
