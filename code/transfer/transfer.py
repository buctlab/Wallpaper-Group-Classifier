#!/usr/bin/env python
# coding: utf-8

# 再训练：树状三网络结果迁移分类，fc一层

# In[1]:


import torch
import sys
sys.path.append('../')
from net.cnn import HLMC


# In[2]:


# pth_path = '../model/'
# pth_files = [pth_path + '1-epoch_18_20_2021-05-14_22.47.10.pkl',
#              pth_path + '2-epoch_8_20_2021-05-15_00.01.15.pkl',
#              pth_path + '3-epoch_15_20_2021-05-15_01.14.48.pkl',
#              pth_path + '4-epoch_16_20_2021-05-15_02.20.30.pkl',
#              pth_path + '5-epoch_13_20_2021-05-15_03.35.12.pkl',
#              pth_path + '6-epoch_11_20_2021-05-16_14.12.00.pkl'
#             ]
# pth_path = '../rotate/output/info/2022-08-08/2/'
pth_files = ['../rotate/output/info/2022-10-08/2022-10-08_01.36.51/epoch_12_20_2022-10-08_01.36.51.pkl',
             '../rotate/output/info/2022-10-15/2022-10-15_04.55.46/epoch_14_20_2022-10-15_04.55.46.pkl',
             '../rotate/output/info/2022-10-07/2022-10-07_18.46.26/epoch_14_20_2022-10-07_18.46.26.pkl']

# 预训练网络模型参数
pretrained_dict = []
for pth in pth_files:
    pretrained_dict.append(torch.load(pth))

# for k, v in pretrained_dict.items():
#     print(k, v)


# In[3]:


# 多模？网络结构
model = HLMC(img_size=224)
model_dict = model.state_dict()
for k, v in model_dict.items():
    print(k)
#     print(k, v)


# In[4]:


# 模型参数导入
for i in range(3):
    ni_dict = {'n'+str(i+1)+'.'+k: v for k, v in pretrained_dict[i].items()}
    model_dict.update(ni_dict)
# for k, v in model_dict.items():
#     print(k, v)
model.load_state_dict(model_dict)


# ## 训练网络
# 只训练最后的fc层

# In[5]:


import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import time,os,csv

from rotate.model_t import ModelT
from rotate.plot_figure import plot_csv_data
from rotate.picture_loader import NoisePictureLoader17

for name, param in model.named_parameters():
    if name == 'fc.weight' or name == 'fc.bias':
        pass
    else:
        param.requires_grad = False # 梯度不更新


# In[7]:


# model.fc.parameters()只对fc进行参数更新
optimizer = optim.SGD(model.fc.parameters(),lr=0.001,momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7,gamma=0.1)


# In[8]:


for name, param in model.named_parameters(): #查看可优化的参数有哪些
     if param.requires_grad:
        print(name)


# In[9]:


# train
def train_run(model, img_size, class_dict, folder, class_num, epochs = 20, batch_size = 40, learning_rate = 0.001):
    data = time.strftime("%Y-%m-%d", time.localtime())
    present = time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime())
    save_path = 'output/{}/{}'.format(data, present)
    print('save_path:', save_path)

    model = ModelT(model=model, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, img_size=img_size,
                   classes=class_dict)
    # 数据
    train_load, test_load = model.data_load_17(class_num, folder, NoisePictureLoader17, data_num) # NoisePictureLoader17:add noise

    if not os.path.exists(save_path):  # 如果不存在保存路径则创建
        os.makedirs(save_path)
    logger = model.get_logger(save_path + '/exp_{}_{}.log'.format(epochs, present))

    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    for i in range(epochs):
        print('train')
        train_loss, ave_train_loss, train_acc = model.train_model(train_load)
        train_loss_list += train_loss
        train_acc_list.append(train_acc)

        print('test')
        test_loss, test_acc = model.test_model(test_load)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        logger.info('Epoch:[{}/{}]\t train:[loss={:.5f}\t acc={:.3f}] test:[loss={:.5f}\t acc={:.3f}]'
                    .format(i, epochs, ave_train_loss, train_acc, test_loss, test_acc))
        torch.save(model.model.state_dict(), save_path + "/epoch_{}_{}_{}.pkl".format(i, epochs, present))

    model.process_figure(train_loss_list, test_loss_list, train_acc_list, test_acc_list, epochs, save_path, present)
    
    f = open(save_path+'/data.csv','w',newline='',encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['train loss'] + train_loss_list) # 写list列表
    csv_writer.writerow(['test loss'] + test_loss_list)
    csv_writer.writerow(['train acc'] + train_acc_list) # 写list列表
    csv_writer.writerow(['test acc'] + test_acc_list)
    f.close()


# In[10]:


img_size = 224
data_num = 4000
class_num = 14 # 14分类

class_dict = {'p1': 0, 'pm': 1, 'pg': 2, 'cm': 3,
              'p2': 4, 'pmm': 5, 'pmg': 6, 'pgg': 7, 'cmm': 6,
              'p4': 8, 'p4m': 9, 'p4g': 9,
              'p3': 10, 'p3m1': 11, 'p31m': 11,
              'p6': 12, 'p6m': 13}

folder =  'D:/jupyter_data/images/output'  # 文件即为folder/classes[idx]/*.png，分类：classes
train_run(model, img_size, class_dict, folder, class_num)


# ## visualization

# In[1]:


from torchsummary import summary


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = Multimodal(img_size=224).to(device)
summary(model, (1,224,224))


# In[ ]:




