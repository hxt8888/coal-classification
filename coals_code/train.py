from model import *
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from dataload import MyDataset,MyDataset1
import csv
from tqdm import tqdm
import codecs
import numpy as np
import time
import matplotlib.pyplot as plt
import torchvision.transforms as tfm
import PIL.Image as Image
from torch.optim.lr_scheduler import  StepLR
import seaborn as sns #导入包


def test_model(test_iter, model, device,name):
    model.eval()
    total_loss = 0.0
    accuracy = 0
    y_true = []
    y_pred = []
    total_test_num = len(test_iter.dataset)
    for i, batch in enumerate(test_iter):
        feature, target = batch
        target = target[:, 0]
        with torch.no_grad():
            target = target.to(device)
            feature = feature.to(device)
            out =model(feature)
            loss = F.cross_entropy(out, target)
            total_loss += loss.item()
            accuracy += (torch.argmax(out, dim=1) == target).sum().item()
            y_true.extend(target.cpu().numpy())
            y_pred.extend(torch.argmax(out, dim=1).cpu().numpy())
            print('>>> batch_{}/{}, Test loss is {}, Accuracy:{} '.format(i, len(test_iter), loss.item(), (
                (torch.argmax(out, dim=1) == target).sum().item()) / target.size(0)))
    print('>>> Test loss:{}, Accuracy:{} \n'.format(total_loss / total_test_num, accuracy / total_test_num))
    score = accuracy_score(y_true, y_pred)
    print(score)
    result = np.concatenate((np.array(y_true).reshape(-1, 1), np.array(y_pred).reshape(-1, 1)), 1)
    np.save('result/%s_result.npy'%name, result)
    from sklearn.metrics import confusion_matrix
    confusion_matrix1 = confusion_matrix(y_true, y_pred)
    print("---------")
    print(y_true)
    print(y_pred)
    print("---------")
    print(confusion_matrix1)
    #混淆矩可视化
    # 画热力图,annot=True 代表 在图上显示 对应的值， fmt 属性 代表输出值的格式，cbar=False, 不显示 热力棒
    x_tick=['1','2','3','4','5','6','7','8','9','10','11']
    y_tick=['1','2','3','4','5','6','7','8','9','10','11']
    # x_tick = ['Ⅰ', 'Ⅱ', 'Ⅲ', 'Ⅳ', 'Ⅴ']
    # y_tick = ['Ⅰ', 'Ⅱ', 'Ⅲ', 'Ⅳ', 'Ⅴ']

    sns.heatmap(confusion_matrix1,fmt='g', cmap='Blues',annot=True,cbar=True,xticklabels=x_tick, yticklabels=y_tick)
    plt.savefig("ft-kt2b.png", dpi=300, bbox_inches='tight')
    plt.show()
    # plt.savefig("E:/000erya308/hanxiaotian/PycharmProjects/coals_code/cnfimg"
    #             , format='jpg'
    #             , bbox_inches='tight'
    #             , pad_inches=0
    #             , dpi=300)
    '''
    '''
    from sklearn.metrics import classification_report
    target_names = ['0', '1', '2', '3', '4','5','6','7','8','9','10']
    # target_names = ['0', '1', '2', '3', '4','5','6','7','8','9','10','11','12','13','14','15','16','17',
    #                 '18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35',
    #                 '36','37','38','39','40','41','42','43','44','45','46']
    print(classification_report(y_true, y_pred, target_names=target_names))


def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("保存文件成功，处理结束")


def train_model(train_iter, dev_iter, model,criterion,optimizer,scheduler, device,name):
    model.train()
    epochs =150
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
            feature, target = batch

            target = target.to(device)
            feature = feature.to(device)
            target = target[:, 0]

            target = target.to(device)
            logit = model(feature)
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
                feature, target = batch
                target = target[:, 0]
                feature = feature.to(device)

                target = target.to(device)
                out = model(feature)
                loss = criterion(out, target)
                total_loss += loss.item()
                accuracy += (torch.argmax(out, dim=1) == target).sum().item()
                lossData2.append(
                    [epoch, j, loss.item(), ((torch.argmax(out, dim=1) == target).sum().item()) / target.size(0)])
                print('>>> batch_{}/{}, Test loss is {}, Accuracy:{} '.format(j, len(test_iter), loss.item(), (
                    (torch.argmax(out, dim=1) == target).sum().item()) / target.size(0)))

        print('>>> Epoch_{}, Valid loss:{}, Accuracy:{} '.format(epoch, total_loss / total_valid_num,
                                                                 accuracy / total_valid_num))
        # scheduler.step()

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
    model = BCNN()
    model=model.to(device)

    # name='BCNN_SE_KTH'
    # name = 'BCNN_SE_KT2b'
    # name = 'BCNN_SE_UIUC'
    # name = 'BCNN_SE_origin_0.75'
    name='BCNN_origin_KT2b_0.5_lp'
    #损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4,betas=(0.5, 0.999))
    scheduler = StepLR(optimizer, step_size=20, gamma=0.8)

    train_tfm = tfm.Compose([tfm.Resize((224)),tfm.RandomCrop(224),
                          tfm.RandomHorizontalFlip(),
                          tfm.RandomRotation(degrees=45),
                          tfm.ToTensor(),
                          tfm.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))])


    test_tfm = tfm.Compose([tfm.Resize((224)),
                            tfm.CenterCrop(224),
                            tfm.ToTensor(),
                            tfm.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))])


    # 加载数据
    traindataset = MyDataset('./split/KT2b_train.txt',train_tfm,False)
    testdataset = MyDataset1('./split/KT2b_test.txt',test_tfm,False)
    #4->9 4->9
    train_iter = DataLoader(dataset=traindataset, batch_size=16, shuffle=True, num_workers=20)
    test_iter = DataLoader(dataset=testdataset, batch_size=16, shuffle=False, num_workers=20)
    # 训练模型
    train_model(train_iter, test_iter, model,criterion,optimizer,scheduler, device,name)
    #测试模型
    # state_dict = torch.load('./modelfile/%s_model.pth'%name)
    # model.load_state_dict(state_dict)
    # time1=time.time()
    # test_model(test_iter, model, device,name)
    # time2 = time.time()
    # print(time2-time1)