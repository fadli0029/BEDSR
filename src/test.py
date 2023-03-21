import os
import utils
import argparse
import numpy as np
from os import listdir

import torch
from torchvision import transforms
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image

from PIL import Image

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res')
parser.add_argument('--scale', type=int, default=2, help='Downsampling scale of input image.')
parser.add_argument('--patch_size', type=int, default=50, help='Patch size.')
parser.add_argument('--input_lr_path', type=str, default='dataset/test_set/lr_x', help='Input path to be used.')
parser.add_argument('--input_hr_path', type=str, default='dataset/test_set/hr/', help='Input path to be used.')
parser.add_argument('--model', type=str, default='checkpoints/model_epoch_150.pth', help='Model to be used.')
parser.add_argument('--output_path', default='results/', type=str, help='Location to save the output images.')
parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')
opt = parser.parse_args()

print(opt)

loader = transforms.Compose([transforms.ToTensor()])

scale = opt.scale
patch_size = opt.patch_size

path_hr = opt.input_hr_path
path_lr = opt.input_lr_path + str(scale) + '/'

image_nums = len([lists for lists in listdir(path_lr) \
        if utils.is_image_file('{}/{}'.format(path_lr, lists))])
print("Number of images:", image_nums)

psnr_avg = 0
for i in listdir(path_lr):
    if utils.is_image_file(i):
        with torch.no_grad():
            img_name = i.split('.')
            img_num = img_name[0]

            # Open original images.
            ori = Image.open('{}{}'.format(path_hr, i)).convert('RGB')
            ori = ori.resize((1200, 800))
            
            # Open low res images.
            lr = Image.open('{}{}'.format(path_lr, i)).convert('RGB')
            lr = lr.resize((1200//scale, 800//scale))
            
            # Get patches.
            imgs = (lr, ori)
            lr_patch, ori_patch, patch_coord = utils.img_to_patch(
                imgs=imgs, scale=scale, patch_size=patch_size)
            
            # Draw bounding box on the original image where patch is taken.
            _, annot_ori = utils.annotated_image((lr, ori), patch_coord)

            # Feed patch to neural net.
            img_to_tensor = ToTensor()
            input = img_to_tensor(lr_patch)
            input = Variable(torch.unsqueeze(input, dim=0).float(), requires_grad=False)

            model = torch.load(opt.model, map_location='cuda:0')
            if opt.cuda:
                model = model.cuda()
                input = input.cuda()

            out = model(input)
            out = out.cpu()
            
            # Convert output tensor `out` to PIL.Image RGB.
            gen_patch = out.data[0].numpy().astype(np.float32)
            gen_patch = gen_patch * 255.
            gen_patch = np.clip(gen_patch, 0., 255.)
            gen_patch = gen_patch.transpose(1, 2, 0)
            gen_patch_img = Image.fromarray(gen_patch.astype(np.uint8))
            gen_patch_img = gen_patch_img.convert('RGB')
            
            # Save patches to folder.
            res_path = opt.output_path+img_num
            if not os.path.exists(res_path):
                os.makedirs(res_path)
            to_pil_image(gen_patch_img).save('{}{}/gen_patch.jpg'.format(opt.output_path, img_num))
            lr_patch.save('{}{}/lr_patch.jpg'.format(opt.output_path, img_num))
            ori_patch.save('{}{}/hr_patch.jpg'.format(opt.output_path, img_num))
            annot_ori.save('{}{}/annot_ori.jpg'.format(opt.output_path, img_num))
            
            # Compute PSNR.
            psnr_val = utils.calc_psnr(loader(gen_patch_img), loader(ori_patch))
            psnr_avg += psnr_val
            print("Test image: {}   ===> PSNR: {}".format(img_num, psnr_val))

psnr_avg = psnr_avg / image_nums
print('AVERAGE PSNR:', psnr_avg)
