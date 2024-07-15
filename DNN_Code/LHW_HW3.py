#导入需要用到的库
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import time

#定义读写文件的函数
def readfile(path,own_label):#后面的参数用来区分训练集和测试集
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir),128,128,3),dtype=np.uint8)
    y = np.zeros((len(image_dir)),dtype=np.uint8)
    for i,file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path,file))
        x[i,:,:] = cv2.resize(img,(128,128))
        if own_label:
            y[i] = int(file.split("_")[0])
    if own_label:
        return x,y
    else:
        return x

#分别读取训练、验证和测试的数据，保存到相应的变量里
workspace_dir = './food-11'  #数据保存的路径
print("Reading data") #提示读取数据
train_x, train_y = readfile(os.path.join(workspace_dir,"training"),True)#读取训练集数据
print("Size of training data = {}".format(len(train_x))) #输出训练集数据的长度
val_x ,val_y = readfile(os.path.join(workspace_dir,"validation"),True) #读取验证集数据
print("Size of validation data = {}".format(len(val_x))) #输出验证集数据的长度
test_x = readfile(os.path.join(workspace_dir,"testing"),False) #读取测试集数据，没有label
print("Size of Testing data = {}".format(len(test_x))) #输出测试集数据长度

#图像增强
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),#随机将图片水平翻转
    transforms.RandomRotation(15),#随机旋转图片
    transforms.ToTensor(),#将图片转化为Tensor，并归一化
])

#测试集图像处理
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

#数据集类处理
class ImgDataset(Dataset):
    def __init__(self,x,y = None,transform = None):
        self.x = x
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self,index):
        X = self.x[index]
        if self.transform is not None:#判断是否有图像变换
            X = self.transform(X) #有，则取变换后的X
        if self.y is not None: #判断是否有y(label)
            Y = self.y[index]
            return X,Y #有，则返回X,Y
        else:
            return X #无y(label),返回X

#初始化batch_size
batch_size = 128

#实例化数据集
train_set = ImgDataset(train_x,train_y,train_transform)
val_set = ImgDataset(val_x,val_y,test_transform)
#载入数据集
train_loader = DataLoader(train_set,batch_size = batch_size,shuffle = True)
val_loader = DataLoader(val_set,batch_size=batch_size,shuffle = False)

#建网络模型
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()

        #建立卷积网络层
        self.cnn = nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),

            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),

            nn.Conv2d(128,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),

            nn.Conv2d(256,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),

            nn.Conv2d(512,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),
        )

        #线性全连接网络
        self.fc = nn.Sequential(
            nn.Linear(512*4*4,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,11)
        )
    #前馈，经过cnn-->view-->fc
    def forward(self,x):
        out = self.cnn(x)
        out = out.view(out.size()[0],-1)
        return self.fc(out)

##训练过程
#model = Classifier().cuda() #实例化，并使用显卡
#loss = nn.CrossEntropyLoss() #loss
#optimizer = torch.optim.Adam(model.parameters(),lr=0.001)#优化
#num_epoch = 30 #训练次数
#
##开始训练
#for epoch in range(num_epoth):
#    #初始化一些数值
#    epoch_start_time = time.time()
#    train_acc = 0.0
#    train_loss = 0.0
#    val_acc = 0.0
#    val_loss = 0.0
#
#    model.train()#启用 BatchNormalization 和 Dropout
#    for i,data in enumerate(train_loader):
#        optimizer.zero_grad() #梯度归零
#        train_pred = model(data[0].cuda()) #前馈
#        batch_loss = loss(train_pred,data[1].cuda()) #损失计算
#        batch_loss.backward() #反馈
#        optimizer.step() #优化
#
#        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(),axis=1)==data[1].numpy())#计算准确率，得到的数和label是否相等
#        train_loss += batch_loss.item() #损失累加
#
#    #在验证集上测试准确率
#    model.eval()#不启用 BatchNormalization 和 Dropout
#    with torch.no_grad():#不求导，节省内存
#        for i,data in enumerate(val_loader):
#            val_pred = model(data[0].cuda()) #前馈
#            batch_loss = loss(val_pred,data[1].cuda()) #计算损失
#
#            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(),axis=1) == data[1].numpy())#计算准确率
#            val_loss += batch_loss.item()#损失累加
#        #打印输出
#        print('[%03d/%03d] %2.2f sec(s) Train Acc:%3.6f Loss:%3.6f | Val Acc:%3.6f loss:%3.6f' %\
#                (epoch + 1,num_epoch,time.time()-epoch_start_time,\
#                train_acc/train_set.__len__(),train_loss/train_set.__len__(),val_acc/val_set.__len__(),val_loss/val_set.__len__()))

train_val_x = np.concatenate((train_x,val_x),axis=0) #合并训练集和验证集的X
train_val_y = np.concatenate((train_y,val_y),axis=0) #合并训练集和验证集的Y
train_val_set = ImgDataset(train_val_x,train_val_y,train_transform) #实例化
train_val_loader = DataLoader(train_val_set,batch_size=batch_size,shuffle=True) #载入数据

model_best = Classifier().cuda() #此处的Classifier应该是你自己调整后，网络结构最好的网络
loss = nn.CrossEntropyLoss() #损失函数
optimizer = torch.optim.Adam(model_best.parameters(),lr=0.001) #优化函数
num_epoch = 30 #训练次数

#开始训练
for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0

    model_best.train()
    for i,data in enumerate(train_val_loader):
        optimizer.zero_grad()
        train_pred = model_best(data[0].cuda())
        batch_loss = loss(train_pred,data[1].cuda())
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

    #输出结果
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
      (epoch + 1, num_epoch, time.time()-epoch_start_time, \
      train_acc/train_val_set.__len__(), train_loss/train_val_set.__len__()))

#测试
test_set = ImgDataset(test_x,transform=test_transform) #测试集实例化
test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False) #载入测试集

model_best.eval() #不启用 BatchNormalization 和 Dropout
prediction = [] #新建列表，用来保存测试结果
with torch.no_grad(): #不求导
    for i,data in enumerate(test_loader):
        test_pred = model_best(data.cuda()) #前馈(预测)
        test_label = np.argmax(test_pred.cpu().data.numpy(),axis=1) #即预测的label
        for y in test_label:
            prediction.append(y)

with open("predict.csv",'w') as f:
    f.write('Id,Categroy\n')
    for i,y in enumerate(prediction):
        f.write('{},{}\n'.format(i,y))
