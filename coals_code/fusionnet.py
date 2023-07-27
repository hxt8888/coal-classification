from model import *
import torch
import torch.nn as nn

#无提升
class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()

        self.bcnn1 = BCNN()
        self.bcnn2 = BCNN()

        state_dict1 = torch.load('./modelfile/BCNN_origin_model.pth')
        self.bcnn1.load_state_dict(state_dict1)

        state_dict2 = torch.load('./modelfile/BCNN_glcm_model.pth')
        self.bcnn2.load_state_dict(state_dict2)


        self.feature1=self.bcnn1.features
        self.feature2 = self.bcnn2.features

        for name, param in self.feature1.named_parameters():
            param.requires_grad = False
        for name, param in self.feature2.named_parameters():
            param.requires_grad = False

        self.weightf1 = nn.Sequential(
            nn.Linear(1024, 1024, bias=False),
            nn.ReLU(),
            nn.Linear(1024, 1024, bias=False),
            nn.ReLU(),
            nn.Linear(1024, 1024, bias=False),
        )

        self.weightf2 = nn.Sequential(
            nn.Linear(1024, 1024//8, bias=False),
            nn.ReLU(),
            nn.Linear(1024//8, 1024//8, bias=False),
            nn.ReLU(),
            nn.Linear(1024//8, 10, bias=False),
        )

        self.classifiers1 = nn.Sequential( nn.Linear(512 ** 2, 5),)
        self.classifiers2 = nn.Sequential( nn.Linear(512 ** 2, 5),)

    def forward(self, x1,x2):
        x10 = self.feature1(x1)
        x20 = self.feature2(x2)

        xf = torch.cat([x10.view(x1.size(0), 512),x20.view(x2.size(0),512)],1)

        wf1=self.weightf1(xf)
        x10=wf1[:,0:512]*x10
        x20=wf1[:,512:]*x20
      
        batch_size = x10.size(0)
        feature_size = x10.size(2)*x10.size(3)
        x1 = x10.view(batch_size, 512, feature_size)
        x1 = (torch.bmm(x1, torch.transpose(x1, 1, 2)) / feature_size).view(batch_size, -1)
        x1 = torch.nn.functional.normalize(torch.sign(x1) * torch.sqrt(torch.abs(x1) + 1e-10))

        batch_size = x20.size(0)
        feature_size = x20.size(2) * x20.size(3)
        x2 = x20.view(batch_size, 512, feature_size)
        x2 = (torch.bmm(x2, torch.transpose(x2, 1, 2)) / feature_size).view(batch_size, -1)
        x2 = torch.nn.functional.normalize(torch.sign(x2) * torch.sqrt(torch.abs(x2) + 1e-10))
        
        
        x1 = self.classifiers1(x1)
        x2 = self.classifiers1(x2)
        wf2=self.weightf2(xf)
        x1=wf2[:,0:5]*x1
        x2=wf2[:,5:]*x2
        return x1+x2

#有提升
class Fusion_improve(nn.Module):
    def __init__(self):
      super(Fusion_improve, self).__init__()

      self.model=Fusion3_1()

      state_dict = torch.load('./modelfile/fusion3_1_model.pth')
      self.model.load_state_dict(state_dict)

      for name, param in self.model.named_parameters():
          param.requires_grad = False

      self.model1=Fusion_1()

      state_dict = torch.load('./modelfile/fusion_1_model.pth')
      self.model1.load_state_dict(state_dict)

      for name, param in self.model1.named_parameters():
          param.requires_grad = False

      self.feature1 = self.model.bcnn1.features
      self.feature2 = self.model.bcnn2.features

      self.weightf1 = nn.Sequential(
          nn.Linear(1024, 1024//8, bias=True),
          nn.ReLU(),
          nn.Linear(1024//8, 1024//8, bias=True),
          nn.ReLU(),
          nn.Linear(1024//8, 2, bias=True),
          nn.Sigmoid())

    def forward(self, x1,x2):
      x10 = self.feature1(x1)
      x20 = self.feature2(x2)
      xf = torch.cat([x10.view(x1.size(0), 512),x20.view(x2.size(0),512)],1)
      w=self.weightf1(xf)
      x=self.model(x1,x2)
      x3=self.model1(x1,x2)
      return w[:,:1]*x + w[:,1:] *x3

#有提升
class Fusion_1(nn.Module):
    def __init__(self):
      super(Fusion_1, self).__init__()

      self.bcnn1 = BCNN()
      self.bcnn2 = BCNN()

      state_dict1 = torch.load('./modelfile/BCNN_origin_model.pth')
      self.bcnn1.load_state_dict(state_dict1)

      state_dict2 = torch.load('./modelfile/BCNN_glcm_model.pth')
      self.bcnn2.load_state_dict(state_dict2)


      self.feature1=self.bcnn1.features
      self.feature2 = self.bcnn2.features

      for name, param in self.bcnn1.named_parameters():
          param.requires_grad = False
      for name, param in self.bcnn2.named_parameters():
          param.requires_grad = False

      self.weightf1 = nn.Sequential(
          nn.Linear(1024, 1024//8, bias=False),
          nn.ReLU(),
          nn.Linear(1024//8, 1024//8, bias=False),
          nn.ReLU(),
          nn.Linear(1024//8, 5, bias=False),
       
      )

      self.weightf2 = nn.Sequential(
          nn.Linear(1024, 1024//8, bias=False),
          nn.ReLU(),
          nn.Linear(1024//8, 1024//8, bias=False),
          nn.ReLU(),
          nn.Linear(1024//8, 5, bias=False),
      
      )

      self.weightf3 = nn.Sequential(
          nn.Linear(1024, 1024//8, bias=False),
          nn.ReLU(),
          nn.Linear(1024//8, 1024//8, bias=False),
          nn.ReLU(),
          nn.Linear(1024//8, 2, bias=False),
   
      )

      self.classifiers = nn.Sequential( nn.Linear(512 ** 2, 5),)

    def forward(self, x1,x2):

      x10 = self.feature1(x1)
      x20 = self.feature2(x2)

      xf = torch.cat([x10.view(x1.size(0), 512),x20.view(x2.size(0),512)],1)

      wf1=self.weightf1(xf)
      wf2=self.weightf2(xf)
      wf3=self.weightf3(xf)

      x1 = self.bcnn1(x1)
      x2 = self.bcnn2(x2)

      x=wf3[:,:1]*wf1*x1+wf3[:,1:]*wf2*x2

      return x
      

#有提升 提升较小
class Fusion1(nn.Module):
    def __init__(self):
      super(Fusion1, self).__init__()

      self.bcnn1 = BCNN()
      self.bcnn2 = BCNN()

      state_dict1 = torch.load('./modelfile/BCNN_origin_model.pth')
      self.bcnn1.load_state_dict(state_dict1)

      state_dict2 = torch.load('./modelfile/BCNN_glcm_model.pth')
      self.bcnn2.load_state_dict(state_dict2)


      self.feature1=self.bcnn1.features
      self.feature2 = self.bcnn2.features

      for name, param in self.feature1.named_parameters():
          param.requires_grad = False
      for name, param in self.feature2.named_parameters():
          param.requires_grad = False

      self.weightf = nn.Sequential(
          nn.Linear(1024, 1024//8, bias=False),
          nn.ReLU(),
          nn.Linear(1024//8, 1024//8, bias=False),
          nn.ReLU(),
          nn.Linear(1024//8, 5, bias=False),
      )

      self.classifiers = nn.Sequential(nn.Linear(1024 ** 2, 5),)
      

    def forward(self, x1,x2):
      x10 = self.feature1(x1)
      x20 = self.feature2(x2)

      xf = torch.cat([x10,x20],1)

      batch_size = xf.size(0)
      feature_size = xf.size(2)*xf.size(3)
      x1 = xf.view(batch_size, 512*2, feature_size)
      x1 = (torch.bmm(x1, torch.transpose(x1, 1, 2)) / feature_size).view(batch_size, -1)
      x1 = torch.nn.functional.normalize(torch.sign(x1) * torch.sqrt(torch.abs(x1) + 1e-10))
     
      x = self.classifiers(x1)
      wf=self.weightf(xf.view(xf.size(0),1024))
      x=wf*x
      return x

#有提升 提升不大
class Fusion2(nn.Module):
    def __init__(self):
        super(Fusion2, self).__init__()

        self.bcnn1 = BCNN()
        self.bcnn2 = BCNN()

        state_dict1 = torch.load('./modelfile/BCNN_origin_model.pth')
        self.bcnn1.load_state_dict(state_dict1)

        state_dict2 = torch.load('./modelfile/BCNN_glcm_model.pth')
        self.bcnn2.load_state_dict(state_dict2)


        self.feature1=self.bcnn1.features
        self.feature2 = self.bcnn2.features

        for name, param in self.feature1.named_parameters():
            param.requires_grad = False
        for name, param in self.feature2.named_parameters():
            param.requires_grad = False

        self.weightf = nn.Sequential(
            nn.Linear(1024, 1024*2, bias=False),
            nn.ReLU(),
            nn.Linear(1024*2, 1024*2, bias=False),
            nn.ReLU(),
            nn.Linear(1024*2, 1024, bias=False),
        )

        self.classifiers = nn.Sequential( nn.Linear(512 ** 2, 5),)

    def forward(self, x1,x2):
        x10 = self.feature1(x1)
        x20 = self.feature2(x2)

        xf = torch.cat([x10.view(x1.size(0), 512),x20.view(x2.size(0),512)],1)

        wf = self.weightf(xf)
        x = wf[:, 0:512] * x10 + wf[:, 512:] * x20

        batch_size = x.size(0)
        feature_size = x.size(2)*x.size(3)
        x = x.view(batch_size, 512, feature_size)
        x = (torch.bmm(x, torch.transpose(x, 1, 2)) / feature_size).view(batch_size, -1)
        x = torch.nn.functional.normalize(torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10))
        x = self.classifiers(x)
        return x
#resnet融合 无提升
class Fusion_res3(nn.Module):
    def __init__(self):
        super(Fusion_res3, self).__init__()

        self.bcnn1 = Resnet18()
        self.bcnn2 = Resnet18()

        # state_dict1 = torch.load('./modelfile/Resnet18_origin_model.pth')
        # self.bcnn1.load_state_dict(state_dict1)

        # state_dict2 = torch.load('./modelfile/Resnet18_glcm_model.pth')
        # self.bcnn2.load_state_dict(state_dict2)


        self.feature1=self.bcnn1.features
        self.feature2 = self.bcnn2.features

        self.classifiers1 = self.bcnn1.classifiers
        self.classifiers2 = self.bcnn2.classifiers

        # for name, param in self.bcnn1.named_parameters():
        #     param.requires_grad = True
        # for name, param in self.bcnn2.named_parameters():
        #     param.requires_grad = True

        self.se=SElayer(512*2)

        self.classifiers = nn.Linear(512*2, 5)

    def forward(self, x1,x2):
        x10 = self.feature1(x1)
        x20 = self.feature2(x2)

        

        x=torch.cat([x10,x20],1)
        # x=self.se(x)

        batch_size = x.size(0)
        x = x.view(batch_size, 512*2)
        x10 = x10.view(batch_size, 512)
        x20 = x20.view(batch_size, 512)
        x=self.classifiers(x)
        x10=self.classifiers1(x10)
        x20=self.classifiers2(x20)
        return x+x10+x20
#有提升
class Fusion3(nn.Module):
    def __init__(self):
        super(Fusion3, self).__init__()

        self.bcnn1 = BCNN()
        self.bcnn2 = BCNN()

        state_dict1 = torch.load('./modelfile/BCNN_origin_model.pth', map_location='cpu')
        self.bcnn1.load_state_dict(state_dict1)

        state_dict2 = torch.load('./modelfile/BCNN_glcm_model.pth', map_location='cpu')
        self.bcnn2.load_state_dict(state_dict2)


        self.feature1=self.bcnn1.features
        self.feature2 = self.bcnn2.features

        for name, param in self.bcnn1.named_parameters():
            param.requires_grad = False
        for name, param in self.bcnn2.named_parameters():
            param.requires_grad = False

        self.weightf = nn.Sequential(
            nn.Linear(1024, 1024//8, bias=False),
            nn.ReLU(),
            nn.Linear(1024//8, 1024//8, bias=False),
            nn.ReLU(),
            nn.Linear(1024//8, 2, bias=False),
        )

        self.classifiers = nn.Sequential( nn.Linear(512 ** 2, 5),)

    def forward(self, x1,x2):

        x10 = self.feature1(x1)
        x20 = self.feature2(x2)

        xf = torch.cat([x10.view(x1.size(0), 512),x20.view(x2.size(0),512)],1)

        wf=self.weightf(xf)
        x1 = self.bcnn1(x1)
        x2 = self.bcnn2(x2)
        x=wf[:,0:1]*x1+wf[:,1:2]*x2
        return x

#有提升


class Fusion30(nn.Module):###融合部分模型
    def __init__(self):
        super(Fusion30, self).__init__()

        self.bcnn1 = BCNN()
        self.bcnn2 = BCNN()

        # self.bcnn1 = Resnet18()
        # self.bcnn2 = Resnet18()

        # state_dict1 = torch.load('./modelfile/BCNN_origin_0.75_model.pth')
        # state_dict1 = torch.load('./modelfile/BCNN_origin_KT_0.5_model.pth')
        state_dict1 = torch.load('./modelfile/BCNN_origin_KT2b_0.5_model.pth')
        # state_dict1 = torch.load('./modelfile/BCNN_origin_UIUC_0.5_model.pth')
        # state_dict1 = torch.load('./modelfile/BCNN_origin_DTD_0.5_model.pth')
        self.bcnn1.load_state_dict(state_dict1)

        # state_dict2 = torch.load('./modelfile/BCNN_glcm_0.75_model.pth')
        # state_dict2 = torch.load('./modelfile/BCNN_glcm_KT_0.5_model.pth')
        state_dict2 = torch.load('./modelfile/BCNN_glcm_KT2b_0.5_model.pth')
        # state_dict2 = torch.load('./modelfile/BCNN_glcm_UIUC_0.5_model.pth')
        # state_dict2 = torch.load('./modelfile/BCNN_glcm_DTD_0.5_model.pth')

        self.bcnn2.load_state_dict(state_dict2)


        self.feature1=self.bcnn1.features
        self.feature2 = self.bcnn2.features

        for name, param in self.bcnn1.named_parameters():
            param.requires_grad = False
        for name, param in self.bcnn2.named_parameters():
            param.requires_grad = False

        self.weightf1 = nn.Sequential(
            nn.Linear(1024, 1024//8, bias=False),
            nn.ReLU(),
            nn.Linear(1024//8, 1024//8, bias=False),
            nn.ReLU(),
            #5->10
            nn.Linear(1024//8,11, bias=False),
            nn.Sigmoid()
        )

        self.weightf2 = nn.Sequential(
            nn.Linear(1024, 1024//8, bias=False),
            nn.ReLU(),
            nn.Linear(1024//8, 1024//8, bias=False),
            nn.ReLU(),
            #5->10
            nn.Linear(1024//8,11, bias=False),
            nn.Sigmoid()
        )
#5->10->25
        self.classifiers = nn.Sequential( nn.Linear(512 ** 2, 11),)

    def forward(self, x1,x2):

        x10 = self.feature1(x1)#把x1,x2传进bcnn计算两个feature为下0，先0
        x20 = self.feature2(x2)

        xf = torch.cat([x10.view(x1.size(0), 512),x20.view(x2.size(0),512)],1)
        #把下，先拉伸为512通道连接
        # xf=torch.cat([x10,x20],1)

        wf1=self.weightf1(xf)#根据连接后的计算权重
        wf2=self.weightf2(xf)

        x1 = self.bcnn1(x1)#把下特征图输入bcnn
        x2 = self.bcnn2(x2)

        x=wf1*x1+wf2*x2
        return x

#有提升
class Fusion3_1(nn.Module):
  def __init__(self):
      super(Fusion3_1, self).__init__()

      self.bcnn1 = BCNN()
      self.bcnn2 = BCNN()

      state_dict1 = torch.load('./modelfile/BCNN_origin_model.pth')
      self.bcnn1.load_state_dict(state_dict1)

      state_dict2 = torch.load('./modelfile/BCNN_glcm_model.pth')
      self.bcnn2.load_state_dict(state_dict2)


      self.feature1=self.bcnn1.features
      self.feature2 = self.bcnn2.features

      for name, param in self.bcnn1.named_parameters():
          param.requires_grad = False
      for name, param in self.bcnn2.named_parameters():
          param.requires_grad = False

      self.weightf1 = nn.Sequential(
          nn.Linear(512, 512//4, bias=False),
          nn.ReLU(),
          nn.Linear(512//4, 512//4, bias=False),
          nn.ReLU(),
          nn.Linear(512//4, 1, bias=False),
      )

      self.weightf2 = nn.Sequential(
          nn.Linear(512, 512//4, bias=False),
          nn.ReLU(),
          nn.Linear(512//4, 512//4, bias=False),
          nn.ReLU(),
          nn.Linear(512//4, 1, bias=False),
      )

      self.classifiers = nn.Sequential( nn.Linear(512 ** 2, 5),)

  def forward(self, x1,x2):

      x10 = self.feature1(x1)
      x20 = self.feature2(x2)

      wf1=self.weightf1(x10.view(x1.size(0), 512))
      wf2=self.weightf2(x20.view(x1.size(0), 512))

      x1 = self.bcnn1(x1)
      x2 = self.bcnn2(x2)
      x=wf1*x1+wf2*x2
      return x

#无提升
class Fusion3_2(nn.Module):
  def __init__(self):
      super(Fusion3_2, self).__init__()

      self.bcnn1 = BCNN()
      self.bcnn2 = BCNN()

      state_dict1 = torch.load('./modelfile/BCNN_origin_model.pth')
      self.bcnn1.load_state_dict(state_dict1)

      state_dict2 = torch.load('./modelfile/BCNN_glcm_model.pth')
      self.bcnn2.load_state_dict(state_dict2)


      self.feature1=self.bcnn1.features
      self.feature2 = self.bcnn2.features

      for name, param in self.bcnn1.named_parameters():
          param.requires_grad = False
      for name, param in self.bcnn2.named_parameters():
          param.requires_grad = False

      self.weightf1 = nn.Sequential(
          nn.Linear(512, 512//4),
          nn.ReLU(),
          nn.Linear(512//4, 512//4),
          nn.ReLU(),
          nn.Linear(512//4, 5),
      )
      self.weightf10 = nn.Sequential(
          nn.Linear(512, 512//4),
          nn.ReLU(),
          nn.Linear(512//4, 512//4),
          nn.ReLU(),
          nn.Linear(512//4, 5),
      )

      self.weightf2 = nn.Sequential(
          nn.Linear(512, 512//4),
          nn.ReLU(),
          nn.Linear(512//4, 512//4),
          nn.ReLU(),
          nn.Linear(512//4, 5),
      )
      self.weightf20 = nn.Sequential(
          nn.Linear(512, 512//4),
          nn.ReLU(),
          nn.Linear(512//4, 512//4),
          nn.ReLU(),
          nn.Linear(512//4, 5),
      )

      self.classifiers = nn.Sequential( nn.Linear(20, 5),)

  def forward(self, x1,x2):

      x10 = self.feature1(x1)
      x20 = self.feature2(x2)

      wf1=self.weightf1(x10.view(x1.size(0), 512))
      wf10=self.weightf1(x20.view(x1.size(0), 512))
      wf2=self.weightf2(x10.view(x1.size(0), 512))
      wf20=self.weightf20(x20.view(x1.size(0), 512))  
      x1 = self.bcnn1(x1)
      x2 = self.bcnn2(x2)
      x=torch.cat([wf1*x1,wf10*x2,wf2*x1,wf20*x2],dim=1)
      x=self.classifiers(x)
      return x
#有提升
class Fusion31(nn.Module):
    def __init__(self):
      super(Fusion31, self).__init__()

      self.bcnn1 = BCNN()
      self.bcnn2 = BCNN()
###
      state_dict1 = torch.load('./modelfile/BCNN_origin_model.pth',map_location='cpu')
      self.bcnn1.load_state_dict(state_dict1)
###
      state_dict2 = torch.load('./modelfile/BCNN_glcm_model.pth',map_location='cpu')
      self.bcnn2.load_state_dict(state_dict2)


      self.feature1=self.bcnn1.features
      self.feature2 = self.bcnn2.features

      for name, param in self.bcnn1.named_parameters():
          param.requires_grad = False
      for name, param in self.bcnn2.named_parameters():
          param.requires_grad = False

      self.weightf1 = nn.Sequential(
          nn.Linear(1024, 1024//8, bias=False),
          nn.ReLU(),
          nn.Linear(1024//8, 1024//8, bias=False),
          nn.ReLU(),
          nn.Linear(1024//8, 25, bias=False),
          nn.Sigmoid()
      )

      self.weightf2 = nn.Sequential(
          nn.Linear(1024, 1024//8, bias=False),
          nn.ReLU(),
          nn.Linear(1024//8, 1024//8, bias=False),
          nn.ReLU(),
          nn.Linear(1024//8, 25, bias=False),
          nn.Sigmoid()
      )

      self.classifiers = nn.Sequential( nn.Linear(512 ** 2, 5),)

    def forward(self, x1,x2):
      x10 = self.feature1(x1)
      x20 = self.feature2(x2)
      xf = torch.cat([x10.view(x1.size(0), 512),x20.view(x2.size(0),512)],1)
      wf1=self.weightf1(xf)
      wf2=self.weightf2(xf)
      x1 = self.bcnn1(x1)
      x2 = self.bcnn2(x2)
      out1=[]
      for i in range(wf1.size(0)):
        out1.append(torch.matmul(x1[i],wf1.view(x10.size(0),5,5)[i]).unsqueeze(0))
      out1=torch.cat(out1,0)
      out2=[]
      for i in range(wf2.size(0)):
        out2.append(torch.matmul(x2[i],wf2.view(x20.size(0),5,5)[i]).unsqueeze(0))
      out2=torch.cat(out2,0)
      x=out1+out2
      return x
if __name__ == '__main__':
    x1=torch.randn((2,3,224,224))
    x2 = torch.randn((2, 3, 224, 224))

    model=Fusion30()
    output=model(x1,x2)
    print(output.shape)