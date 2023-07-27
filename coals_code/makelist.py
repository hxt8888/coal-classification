import os

path_train='./Coal/train'
path_test='./Coal/test'


# path_train='./KTH-TIPS/train'
# path_test='./KTH-TIPS/test'

# path_train='./DTD/train'
# path_test='./DTD/test'


# f=open('./split/train.txt','w')
f=open('./split/Coal_train_1.txt','w')
name_class=[]
for i,name_train in enumerate(os.listdir(path_train)):
    name_class.append(name_train)
    train=os.listdir(path_train+'/'+name_train)
    for t in train:
        info='%s,%d\n'%(path_train+'/'+name_train+'/'+t,i)
        f.writelines(info)
f.close()


f=open('./split/Coal_test_1.txt','w')
for i,name_test in enumerate(name_class):
    test=os.listdir(path_test+'/'+name_test)
    for t in test:
        info='%s,%d\n'%(path_test+'/'+name_test+'/'+t,i)
        f.writelines(info)
f.close()