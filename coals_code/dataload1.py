import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class MyDataset(Dataset):
    def __init__(self,filename,transform,glem=False):
        self.name=filename
        self.frame_list = list()
        self.transform=transform
        self.glem=glem
        if glem:
            with open(self.name) as split_f:
                data = split_f.readlines()
                for line in data:
                    self.frame_list.append(line.replace('coals data/train','glcm_coals/images'))
        else:
            with open(self.name) as split_f:
                data = split_f.readlines()
                for line in data:
                    self.frame_list.append(line)
        self.len = len(self.frame_list)

    def __getitem__(self, index):
        data=self.frame_list[index]
        data_info = data.split(',')
        path = data_info[0]
        x_data = Image.open(path).convert('RGB')
        # if self.glem:
        #     width, height = x_data.size
        #     x_data = x_data.crop((17,2,width,height-17))
        if self.transform != None:
           self.x_data=self.transform(x_data)
        y_data=np.array([int(data_info[1])])
        self.y_data = torch.from_numpy(y_data).long()
        return self.x_data,self.y_data
    def __len__(self):
        return self.len

class MyDataset1(Dataset):
    def __init__(self,filename,transform,glem=False):
        self.name=filename
        self.frame_list = list()
        self.transform=transform
        self.glem=glem
        if glem:
            with open(self.name) as split_f:
                data = split_f.readlines()
                for line in data:
                    self.frame_list.append(line.replace('coals data/test','glcm_coals/images'))
        else:
            with open(self.name) as split_f:
                data = split_f.readlines()
                for line in data:
                    self.frame_list.append(line)
        self.len = len(self.frame_list)

    def __getitem__(self, index):
        data=self.frame_list[index]
        data_info = data.split(',')
        path = data_info[0]
        x_data = Image.open(path).convert('RGB')
        # if self.glem:
        #     width, height = x_data.size
        #     x_data = x_data.crop((17,2,width,height-17))
        if self.transform != None:
           self.x_data=self.transform(x_data)
        y_data=np.array([int(data_info[1])])
        self.y_data = torch.from_numpy(y_data).long()
        return self.x_data,self.y_data
    def __len__(self):
        return self.len


class MyDataset2(Dataset):
    def __init__(self,filename,transform):
        self.name=filename
        self.frame_list = list()
        self.glem_list = list()
        self.transform=transform
        with open(self.name) as split_f:
                data = split_f.readlines()
                for line in data:
                    self.glem_list.append(line.replace('coals data/train','glcm_coals/images'))
                    self.frame_list.append(line)
        self.len = len(self.frame_list)

    def __getitem__(self, index):
        data1=self.frame_list[index]
        data_info1 = data1.split(',')
        path1 = data_info1[0]
        x_data1 = Image.open(path1).convert('RGB')

        data2 = self.glem_list[index]
        data_info2 = data2.split(',')
        path2 = data_info2[0]
        x_data2 = Image.open(path2).convert('RGB')
        # width, height = x_data2.size
        # x_data2 = x_data2.crop((17,2,width,height-17))
        if self.transform != None:
           self.x_data1=self.transform(x_data1)
           self.x_data2 = self.transform(x_data2)
        y_data=np.array([int(data_info1[1])])
        self.y_data = torch.from_numpy(y_data).long()
        return self.x_data1,self.x_data2,self.y_data

    def __len__(self):
        return self.len


class MyDataset2_1(Dataset):
    def __init__(self,filename,transform):
        self.name=filename
        self.frame_list = list()
        self.glem_list = list()
        self.transform=transform
        with open(self.name) as split_f:
                data = split_f.readlines()
                for line in data:
                    self.glem_list.append(line.replace('coals data/test','glcm_coals/images'))
                    self.frame_list.append(line)
        self.len = len(self.frame_list)

    def __getitem__(self, index):
        data1=self.frame_list[index]
        data_info1 = data1.split(',')
        path1 = data_info1[0]
        x_data1 = Image.open(path1).convert('RGB')

        data2 = self.glem_list[index]
        data_info2 = data2.split(',')
        path2 = data_info2[0]
        x_data2 = Image.open(path2).convert('RGB')
        # width, height = x_data2.size
        # x_data2 = x_data2.crop((17,2,width,height-17))
        if self.transform != None:
           self.x_data1=self.transform(x_data1)
           self.x_data2 = self.transform(x_data2)
        y_data=np.array([int(data_info1[1])])
        self.y_data = torch.from_numpy(y_data).long()
        return self.x_data1,self.x_data2,self.y_data

    def __len__(self):
        return self.len