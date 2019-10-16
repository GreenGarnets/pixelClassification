from typing import Tuple, List, Optional, Union
import torchvision

import math
import sys

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from math import ceil

from PIL import Image
import transforms as T

import ResnetColor
from ColorMaskDataset import ColorMaskDataset
import utils

def eval(model, filename):
    img = Image.open(filename).convert("RGB") 
    input = ToTensor()(img).unsqueeze(0)
    (width, height) = img.size
    width, height = int(ceil(width / 4)), int(ceil(height / 4))
    out = model(input)
    out = F.softmax(out,dim = 1)

    outputImg = Image.new("RGB", (width,height),(0,0,0))
    output = outputImg.load()
    for x in range(out.shape[0]) :
        maxC = -1
        c = 0
        #print(out[x])
        if out[x,0] > maxC :
            maxC = out[x,0]
            c = 0  
        if out[x,1] > maxC :
            maxC = out[x,1]
            c = 1
        if out[x,2] > maxC : 
            maxC = out[x,2]
            c = 2
        if out[x,3] > maxC : 
            maxC = out[x,3]
            c = 3
        
        #print(c)
        if c == 1 :
            output[x % width,int(x / width)] = (255,0,0)
        elif c == 2 : 
            output[x % width,int(x / width)] = (0,255,0)
        elif c == 3 : 
            output[x % width,int(x / width)] = (0,0,255)
        elif c == 0 :
            output[x % width,int(x / width)] = (0,0,0)

    outputImg.save(filename + "_.png")
    outputImg = outputImg.resize((width*4,height*4))
    outputImg.save(filename + "__.png")

def main():
    
    dataset = ColorMaskDataset("./dataset/")
    train_loader = DataLoader(dataset=dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=0)

    model = ResnetColor.ResnetColor()
    #model.load_state_dict(torch.load("./model.pth"))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # move model to the right device
    model.to(device)
    epochs = 1600

    # construct an optimizer
    CrossEntropyLoss = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001,
                                momentum=0.9, weight_decay=0.0005)
    

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=7,
                                                    gamma=0.1)    

    for epoch in range(epochs) :
        #try :
        running_loss = 0.0
        for index, (input, target, imgName, maskName) in enumerate(train_loader) :
            #img = Image.open("./dataset/Image/5.jpg").convert("RGB")
            #mask = Image.open("./dataset/Mask/5_mask.png").convert("RGB")

            input = input.to(device)
            target = target.to(device)
            target = target.view(-1)
            optimizer.zero_grad()
            #print(target.shape)
            
            pred = model(input)
            #print(pred.shape)
            #print(target.shape)

            loss = CrossEntropyLoss(pred, target)

            #loss_dict_reduced = utils.reduce_dict(loss)
            #losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            #loss_value = losses_reduced.item()

            if not math.isfinite(loss):
                print("Loss is {}, stopping training".format(loss))
                print(loss)
                sys.exit(1)

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            #if epoch+1 == 3000 :
                #pred = F.softmax(pred,dim = 1)
                #print(pred[143])
                #print(target)

        epoch_loss = running_loss / len(train_loader.dataset)
        print('epoch : {} \tloss : {:.3}'.format(epoch+1, epoch_loss))
        if epoch+1 % 100 == 0 : 
            torch.save(model.state_dict(),"./model.pth")
        #except ValueError :
            #print(imgName, maskName)
    
    torch.save(model.state_dict(),"./model.pth")
    device = torch.device("cpu")
    model.to(device)

    eval(model, "./dataset/test.jpg")
    eval(model, "./dataset/test1.jpg")
    eval(model, "./dataset/test2.jpg")

if __name__ == "__main__":
    main()

'''
# output -> maskImg, 차후 데이터로더로 이동
outputImg = Image.new("RGB", (out.shape[2],out.shape[1]),(255,255,255))
output = outputImg.load()
for x in range(out.shape[2]) :
    for y in range(out.shape[1]) :
        maxC = -1
        c = 0
        if out[0,y,x,0] > maxC :
            maxC = out[0,y,x,0]
            c = 0
        if out[0,y,x,1] > maxC :
            maxC = out[0,y,x,1]
            c = 1
        if out[0,y,x,2] > maxC : 
            maxC = out[0,y,x,2]
            c = 2
        if out[0,y,x,3] > maxC : 
            maxC = out[0,y,x,3]
            c = 3
        
        #print(c)
        if c == 1 :
            output[x,y] = (255,0,0)
        elif c == 2 : 
            output[x,y] = (0,255,0)
        elif c == 3 : 
            output[x,y] = (0,0,255)
        elif c == 0 :
            output[x,y] = (0,0,0)

outputImg = outputImg.resize((out.shape[2] * 16, out.shape[1] * 16))
outputImg.save("./dataset/test.png")

# output -> maskImg, 차후 데이터로더로 이동, 변환된 상태로 target에 넣어야함
mask = mask.resize((int(mask.width/ 16), int(mask.height / 16)))
mask = mask.resize((int(mask.width* 16), int(mask.height * 16)))
mask.save("./dataset/test_mask.png")

'''
'''
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)'''
