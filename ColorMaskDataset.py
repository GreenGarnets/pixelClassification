
import torch
import numpy as np

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from math import ceil

import os
import numpy as np
import torch
from PIL import Image

class ColorMaskDataset(Dataset):    

    def targetToTensor(self, maskImg) :
        (width, height) = maskImg.size
        maskImg = maskImg.resize((ceil(width/16),ceil(height/16)))
        (width, height) = maskImg.size
        #print(width,height)

        target = torch.empty(width*height, dtype=torch.long)
        #print(target.size())

        for y in range(0, height): 
            for x in range(0, width): 
                r, g, b = maskImg.getpixel((x, y))
                #print(str(j) + ' '  + str(i) + ' ' + str((i*width)+j) + ' ' + str(r) + ',' + str(g) + ',' + str(b))
                if r > 140 :
                    target[(y*width)+x] = 1
                elif b > 140 :
                    target[(y*width)+x] = 2
                elif g > 140 :
                    target[(y*width)+x] = 3
                else :
                    target[(y*width)+x] = 0
        
        '''
        for i in range(0, height): 
            for j in range(0, width): 
                print(int(target[(i*width)+j]), end = ' ')
            print('')
        '''

        '''        
        # Target Mask Output Test
        outputImg = Image.new("RGB", (width,height),(0,0,0))
        output = outputImg.load()
        for x in range(0,width * height) :
            print(x / width,x % width)
            if target[x] == 1 :
                output[x % width,int(x / width)] = (255,0,0)
            elif target[x] == 2 : 
               output[x % width,int(x / width)] = (0,255,0)
            elif target[x] == 3 : 
                output[x % width,int(x / width)] = (0,0,255)
            elif target[x] == 0 :
                output[x % width,int(x / width)] = (0,0,0)

        #outputImg = outputImg.resize((width*16,height*16))
        outputImg.save("./dataset/TargetTest.png")
        
        exit()
        '''

        return target    

    # Initialize your data, download, etc.
    def __init__(self, root):
        # load all image files, sorting them to
        # ensure that they are aligned
        self.root = root
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Image"))))
        print(self.imgs)
        self.len = len(self.imgs)

    def __getitem__(self, index):
        # load images ad masks
        img_path = os.path.join(self.root + "Image/", self.imgs[index])
        mask_path = os.path.join(self.root + "Mask/", self.imgs[index].replace(".jpg","_mask.png"))
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        
        input = ToTensor()(img).unsqueeze(0)
        target = self.targetToTensor(mask)

        input = input.squeeze()
        
        #print(input.shape)

        return input, target, img_path, mask_path

    def __len__(self):
        return self.len