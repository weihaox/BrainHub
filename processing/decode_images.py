#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   decode_images.py
@Time    :   2024/03/13 15:52:38
@Author  :   Weihao Xia 
@Version :   1.0
@Desc    :   None
'''

import os
import torch
import torchvision.transforms.functional as F

data_path = 'caption/all_images.pt'
save_path = 'caption/test_images'

if not os.path.exists(save_path):
    os.makedirs(save_path)

images = torch.load(data_path)

for idx, image in enumerate(images):
    print (f'Processing image {idx}')
    image_pil = F.to_pil_image(image.squeeze(0))
    image_pil.save(os.path.join(save_path, f'{idx}.png'))