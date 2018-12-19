import torch
from torch import nn
#from torch.utils.data import DataLoader
#from torch.autograd import Variable
from torchvision import transforms, datasets, models
#import time
import os
import numpy as np
#import torchvision
#import matplotlib.pyplot as plt
from PIL import Image


transform2= transforms.Compose([
        # 图片大小缩放 统一图片格式
        transforms.Resize(256),
        # 以中心裁剪
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
    ])

class CNN(nn.Module):
    def __init__(self, in_dim, n_class):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(True),
            nn.Conv2d(in_dim, 16, 7), # 224 >> 218
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 218 >> 109
            nn.ReLU(True),
            nn.Conv2d(16, 32, 5),  # 105
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5),  # 101
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 101 >> 50
            nn.Conv2d(64, 128, 3, 1, 1),  #
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(3),  # 50 >> 16
        )
        self.fc = nn.Sequential(
            nn.Linear(128*16*16, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(True),
            nn.Linear(120, n_class))
    def forward(self, x):
        out = self.cnn(x)
        out = self.fc(out.view(-1, 128*16*16))
        return out

#输入3层rgb ，输出 分类 2
model = CNN(3, 2)

if os.path.exists('./person_scene_cnn.pth'): #############继续训练
    model.load_state_dict(torch.load('./person_scene_cnn.pth'))

model.train(False)

def Recognize_one(path):
    img_path = path
    #r"D:\Documents\python_code\test_proj\person_scene_data\train\scene\scene_6.jpg"
    img = Image.open(img_path).convert('RGB')  # 读取图像
    input=transform2(img)

    input_4=torch.from_numpy(np.array(input.numpy(),ndmin=4))

    outputs = model(input_4)#数据输入
    preds = torch.max(outputs.data, 1)[1]
    if preds.numpy()==[0]:
        return "人物照"
    else:
        return "风景照"


