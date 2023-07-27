import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import *


class BCNN(nn.Module):#主要模型
    def __init__(self):
        super(BCNN, self).__init__()
        model=resnet18(pretrained=True)
        self.features = nn.Sequential(model.conv1,
                      model.bn1,
                      model.relu,
                      model.maxpool,
                      model.layer1,
                      model.layer2,
                      model.layer3,
                      model.layer4,
                      model.avgpool)

        self.classifiers = nn.Sequential(nn.Linear(512 ** 2,11),)
        # self.classifiers = nn.Sequential(nn.Linear(512 ** 2,47), )

    def forward(self, x):
        x = self.features(x)#[2,512,1,1]
        batch_size = x.size(0)#batch_size=2
        channel_size=x.size(1)
        feature_size = x.size(2)*x.size(3)#feature_size=1
        x = x.view(batch_size, 512, feature_size)#origin 512 res101_>2048
        x = (torch.bmm(x, torch.transpose(x, 1, 2)) / feature_size).view(batch_size, -1)#[2,512,1]
        x = torch.nn.functional.normalize(torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10))#[2,262144]
        x = self.classifiers(x)#[2,11]
        return x

class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        model=resnet18(pretrained=True)
        self.features = nn.Sequential(model.conv1,
                      model.bn1,
                      model.relu,
                      model.maxpool,
                      model.layer1,
                      model.layer2,
                      model.layer3,
                      model.layer4,
                      model.avgpool)

        #5->10
        self.classifiers = nn.Sequential(nn.Linear(512,11),)

    def forward(self, x):
        x = self.features(x)
        batch_size = x.size(0)
        x = x.view(batch_size, 512)
        x = self.classifiers(x)
        return x

class Resnet18_SE(nn.Module):
    def __init__(self):
        super(Resnet18_SE, self).__init__()
        model=resnet18_se(pretrained=False)
        self.features = nn.Sequential(model.conv1,
                      model.bn1,
                      model.relu,
                      model.maxpool,
                      model.layer1,
                      model.layer2,
                      model.layer3,
                      model.layer4,
                      model.avgpool)

        self.classifiers = nn.Sequential(nn.Linear(512,5),)

    def forward(self, x):
        x = self.features(x)
        batch_size = x.size(0)
        x = x.view(batch_size, 512)
        x = self.classifiers(x)
        return x

class BCNN_SE(nn.Module):
    def __init__(self):
        super(BCNN_SE, self).__init__()
        model=resnet18_se(pretrained=False)
        self.features = nn.Sequential(model.conv1,
                      model.bn1,
                      model.relu,
                      model.maxpool,
                      model.layer1,
                      model.layer2,
                      model.layer3,
                      model.layer4,
                      model.avgpool)

        self.classifiers = nn.Sequential(nn.Linear(512 ** 2,5),)
        # self.classifiers = nn.Sequential(nn.Linear(512 ** 2,47), )

    def forward(self, x):
        x = self.features(x)#[2,512,1,1]
        batch_size = x.size(0)#batch_size=2
        channel_size=x.size(1)
        feature_size = x.size(2)*x.size(3)#feature_size=1
        x = x.view(batch_size, 512, feature_size)#origin 512 res101_>2048
        x = (torch.bmm(x, torch.transpose(x, 1, 2)) / feature_size).view(batch_size, -1)#[2,512,1]
        x = torch.nn.functional.normalize(torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10))#[2,262144]
        x = self.classifiers(x)#[2,11]
        return x
