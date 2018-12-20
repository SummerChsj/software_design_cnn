import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms, datasets, models
import time
import os
# import numpy as np
# import torchvision
# import matplotlib.pyplot as plt

#person=0 scene=1

BATCH_SIZE = 50 ############if data is larger,chang it!
LR = 0.001
EPOCHES = 10#################增加epoch，提高准确率

USE_GPU = True #######if you use gpu,make it to be True
if USE_GPU:
    gpu_status = torch.cuda.is_available()
else:
    gpu_status = False

data_transforms = {
    'train': transforms.Compose([
        # 随机切成224x224 大小图片 统一图片格式
        transforms.RandomResizedCrop(224),
        # 图像翻转
        transforms.RandomHorizontalFlip(),
        # totensor 归一化(0,255) >> (0,1)   normalize channel=（channel-mean）/std
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
    ]),
    "val" : transforms.Compose([
        # 图片大小缩放 统一图片格式
        transforms.Resize(256),
        # 以中心裁剪
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
    ])
}

data_dir = 'D:/Documents/python_code/test_proj/person_scene_data/'  ###############your path
# trans data
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# load data
data_loaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True) for x in ['train', 'val']}

data_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(data_sizes, class_names)

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
print(model)

if gpu_status:
    net = model.cuda()
    print("使用gpu")
else:
    print("使用cpu")

loss_f = nn.CrossEntropyLoss()
#optimizer= optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99))#############可以改变优化器，提高速度
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)#######改变学习率

start_time = time.time()

# if os.path.exists('./person_scene_cnn.pth'): #############继续训练
#     model.load_state_dict(torch.load('./person_scene_cnn.pth'))

best_model_wts = model.state_dict()
best_acc = 0.0
train_loss, test_loss, train_acc, test_acc, time_p = [], [], [], [], []

for epoch in range(EPOCHES):

    scheduler.step()
    model.train(True)

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for data in data_loaders['train']:
        inputs, labels = data

        if gpu_status:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        optimizer.zero_grad()

        outputs = model(inputs)#数据输入

        preds = torch.max(outputs.data, 1)[1]
        loss = loss_f(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()*len(labels)

        running_corrects += torch.sum(preds == labels.data)


    epoch_loss = running_loss / data_sizes['train']
    epoch_acc = running_corrects.numpy() / data_sizes['train']

    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)

    # if phase == 'val' and epoch_acc > best_acc:
    #     best_acc = epoch_acc
    #     best_model_wts = model.state_dict()

    time_elapsed=time.time() - start_time
    time_p.append(time_elapsed)
    # print("[{}/{}] train_loss:{:.3f}||test_loss:{:.3f}||train_acc:{:.3f}||test_acc:{:.3f}||time passed:{:.0f}m {:.0f}s".format(epoch+1, EPOCHES,
    #                                            train_loss[-1], test_loss[-1], train_acc[-1], test_acc[-1],time_elapsed // 60, time_elapsed % 60))
    print("[{}/{}] train_loss:{:.3f}||train_acc:{:.3f}||time passed:{:.0f}m {:.0f}s".format(epoch + 1, EPOCHES,
                                            train_loss[-1],  train_acc[-1], time_elapsed // 60, time_elapsed % 60))


start_time=time.time()

model.train(False)
running_loss = 0.0
running_corrects = 0

# Iterate over data.
for data in data_loaders['val']:
    inputs, labels = data


    if gpu_status:
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
    else:
        inputs, labels = Variable(inputs), Variable(labels)

    #optimizer.zero_grad()

    outputs = model(inputs)#数据输入

    preds = torch.max(outputs.data, 1)[1]
    loss = loss_f(outputs, labels)

    running_loss += loss.item()*len(labels)

    running_corrects += torch.sum(preds == labels.data)


recognize_loss =running_loss / data_sizes['val']
recognize_acc = running_corrects.numpy() / data_sizes['val']
recognize_time=time.time()-start_time

print("recognize_loss: %s||recognize_acc: %s||recognize_time: %sm %ss" %(recognize_loss,recognize_acc,recognize_time//60,recognize_time%60))

torch.save(model.state_dict(), "./person_scene_cnn.pth")
