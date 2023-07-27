from model import *
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from dataload import MyDataset2_1
import csv
from tqdm import tqdm
import codecs
import numpy as np
import time
import matplotlib.pyplot as plt
import torchvision.transforms as tfm
import PIL.Image as Image
from torch.optim.lr_scheduler import  StepLR


def test_model(test_iter, model1,model2, device):
    model1.eval()
    model2.eval()
    total_loss = 0.0
    accuracy = 0
    y_true = []
    y_pred = []
    total_test_num = len(test_iter.dataset)
    for i, batch in enumerate(test_iter):
        feature1,feature2, target = batch
        target = target[:, 0]
        with torch.no_grad():
            target = target.to(device)
            feature1 = feature1.to(device)
            feature2 = feature2.to(device)
            out1 =model1(feature1)
            out2 = model2(feature2)
            out = out1+out2
            accuracy += (torch.argmax(out, dim=1) == target).sum().item()
            y_true.extend(target.cpu().numpy())
            y_pred.extend(torch.argmax(out, dim=1).cpu().numpy())
            print('>>> batch_{}/{}, Test Accuracy:{} '.format(i, len(test_iter),  (
                (torch.argmax(out, dim=1) == target).sum().item()) / target.size(0)))
    print('>>> Test loss:{}, Accuracy:{} \n'.format(total_loss / total_test_num, accuracy / total_test_num))
    score = accuracy_score(y_true, y_pred)
    print(score)
    result = np.concatenate((np.array(y_true).reshape(-1, 1), np.array(y_pred).reshape(-1, 1)), 1)
    np.save('result/0.5add_result.npy', result)
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_true, y_pred)
    print(confusion_matrix)
    from sklearn.metrics import classification_report
    target_names = ['0', '1', '2', '3', '4']
    print(classification_report(y_true, y_pred, target_names=target_names))


def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("保存文件成功，处理结束")


def train_model(train_iter, dev_iter, model,criterion,optimizer,scheduler, device,name):
    model.train()
    epochs = 200
    print('training...')
    lossData1 = [[]]
    lossData2 = [[]]
    acc = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        accuracy = 0
        total_train_num = len(train_iter.dataset)
        for i, batch in enumerate(train_iter):
            feature1,feature2, target = batch

            target = target.to(device)
            feature1 = feature1.to(device)
            feature2 = feature2.to(device)
            target = target[:, 0]

            target = target.to(device)
            logit = model(feature1,feature2)
            loss = criterion(logit, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            accuracy += (torch.argmax(logit, dim=1) == target).sum().item()
            print('>>> batch_{}/{}, Train loss is {}, Accuracy:{} '.format(i, len(train_iter), loss.item(), (
                (torch.argmax(logit, dim=1) == target).sum().item()) / target.size(0)))
            lossData1.append(
                [epoch, i, loss.item(), ((torch.argmax(logit, dim=1) == target).sum().item()) / target.size(0)])
        print('>>> Epoch_{}, Train loss is {}, Accuracy:{} '.format(epoch, total_loss / total_train_num,
                                                                    accuracy / total_train_num))

        model.eval()
        total_loss = 0.0
        accuracy = 0
        total_valid_num = len(dev_iter.dataset)
        for j, batch in enumerate(dev_iter):
            with torch.no_grad():
                feature1,feature2, target = batch
                target = target[:, 0]
                feature1 = feature1.to(device)
                feature2 = feature2.to(device)

                target = target.to(device)
                out = model(feature1,feature2)
                loss = criterion(out, target)
                total_loss += loss.item()
                accuracy += (torch.argmax(out, dim=1) == target).sum().item()
                lossData2.append(
                    [epoch, j, loss.item(), ((torch.argmax(out, dim=1) == target).sum().item()) / target.size(0)])
                print('>>> batch_{}/{}, Test loss is {}, Accuracy:{} '.format(j, len(test_iter), loss.item(), (
                    (torch.argmax(out, dim=1) == target).sum().item()) / target.size(0)))

        print('>>> Epoch_{}, Valid loss:{}, Accuracy:{} '.format(epoch, total_loss / total_valid_num,
                                                                 accuracy / total_valid_num))
        scheduler.step()

        if accuracy / total_valid_num > acc:
            saveModel(model, name)
            acc = accuracy / total_valid_num
        data_write_csv("./csv/%s_loss.csv"%name, lossData1)
        data_write_csv("./csv/%s_val_loss.csv"%name, lossData2)


def saveModel(model, name):
    torch.save(model.state_dict(), 'modelfile/' + name + '_model.pth')


if __name__ == '__main__':
    # 加载模型
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model1 = BCNN()
    model1=model1.to(device)
    state_dict1 = torch.load('./modelfile/resnet18_model.pth')
    model1.load_state_dict(state_dict1)

    model2 = BCNN()
    model2 = model2.to(device)
    state_dict2 = torch.load('./modelfile/resnet18_glcm_model.pth')
    model2.load_state_dict(state_dict2)


    #损失函数和优化器
    test_tfm = tfm.Compose([tfm.Resize((224)),
                            tfm.CenterCrop(224),
                            tfm.ToTensor(),
                            tfm.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))])


    # 加载数据
    testdataset = MyDataset2_1('./split/test.txt',test_tfm)
    test_iter = DataLoader(dataset=testdataset, batch_size=2, shuffle=False, num_workers=4)
    # 训练模型
    test_model(test_iter, model1,model2, device)