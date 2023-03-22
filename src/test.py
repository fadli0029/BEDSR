import os
import utils
import argparse
import numpy as np
from os import listdir

import torch
from torch.autograd import Variable

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

scale = opt.scale
patch_size = opt.patch_size
path_hr = opt.input_hr_path
path_lr = opt.input_lr_path + str(scale) + '/'

image_nums = len([lists for lists in listdir(path_lr) \
        if utils.is_image_file('{}/{}'.format(path_lr, lists))])
print("Number of images:", image_nums)

psnrs = {}
psnr_avg = 0
for i in listdir(path_lr):
    if utils.is_image_file(i):
        with torch.no_grad():
            img_name = i.split('.')
            img_num = img_name[0]

            # Open original images.
            ori = Image.open('{}{}'.format(path_hr, i)).convert('RGB')
            
            # Open low res images.
            lr = Image.open('{}{}'.format(path_lr, i)).convert('RGB')
            
            # Get patches.
            imgs = (lr, ori)
            lr_patch, ori_patch, patch_coord = utils.img_to_patch(
                imgs=imgs, scale=scale, patch_size=patch_size)

            # Draw bounding box on the original image where patch is taken.
            _, annot_ori = utils.annotated_image((lr, ori), patch_coord)

            # Feed patch to neural net.
            input = torch.from_numpy(np.array(lr_patch)).permute(2, 0, 1).float()
            input = Variable(torch.unsqueeze(input, dim=0).float(), requires_grad=False)
            # `input` shape: (1, 3, patch_size, patch_size)

            # Load model
            model = torch.load(opt.model, map_location='cuda:0')

            # Put model to cuda.
            if opt.cuda:
                model = model.cuda()
                input = input.cuda()

            out = model(input)
            out = out.cpu()
            
            # Convert output tensor `out` to PIL.Image RGB.
            gen_patch_img = utils.output_to_image(out)
            
            # Save patches to folder.
            res_path = opt.output_path+img_num
            if not os.path.exists(res_path):
                os.makedirs(res_path)
            gen_patch_img.save('{}{}/gen_patch.jpg'.format(opt.output_path, img_num))
            lr_patch.save('{}{}/lr_patch.jpg'.format(opt.output_path, img_num))
            ori_patch.save('{}{}/hr_patch.jpg'.format(opt.output_path, img_num))
            annot_ori.save('{}{}/annot_ori.jpg'.format(opt.output_path, img_num))
            
            # Compute PSNR.
            psnr_val = utils.calc_psnr(out[0], utils.image_to_tensor(ori_patch))
            psnrs[img_num] = psnr_val
            psnr_avg += psnr_val
            print("Test image: {}   ===> PSNR: {}".format(img_num, psnr_val))

psnr_avg = psnr_avg / image_nums
psnrs = sorted(psnrs.items(), key=lambda x:x[1], reverse=True)
print('AVERAGE PSNR:', psnr_avg)
print('Images with best PSNR:\n')
print(psnrs[:3])
