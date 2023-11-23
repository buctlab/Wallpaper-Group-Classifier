import time
import os
import sys
import csv
import pandas as pd
import torch
from torchvision import transforms

sys.path.append('../')  # 调用上一级文件夹下方法
from rotate.model_t import ModelT, EarlyStopping
from rotate.plot_figure import plot_csv_data

from net.cnn import CNN5
from rotate.picture_loader import NoisePictureLoader17


def train_run(model, img_size, class_dict, folder, class_num, pipelines, epochs=20, batch_size=40, learning_rate=0.001):
    data = time.strftime("%Y-%m-%d", time.localtime())
    present = time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime())
    save_path = 'output/info/{}/{}'.format(data, present)
    print('save_path:', save_path, '\n')
    print(present)

    model = ModelT(model=model, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, img_size=img_size,
                   classes=class_dict, pipelines=pipelines)
    # 数据，data_load_kind='mix'混杂数据
    train_load, test_load = model.data_load_17(class_num, folder, NoisePictureLoader17, data_num,
                                               data_load_kind='normal')  # 'normal' 'mix' NoisePictureLoader17:add noise

    if not os.path.exists(save_path):  # 如果不存在保存路径则创建
        os.makedirs(save_path)
    logger = model.get_logger(save_path + '/exp_{}_{}.log'.format(epochs, present))
    logger.info('only mini_rotate, crop_left_upper, without noise')

    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    early_stopping = EarlyStopping(patience=5)
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

        early_stopping(test_loss)
        # 达到早停止条件时，early_stop会被置为True
        if early_stopping.early_stop:
            print("Early stopping")
            break  # 跳出迭代，结束训练

    model.process_figure(train_loss_list, test_loss_list, train_acc_list, test_acc_list, epochs, save_path, present)

    f = open(save_path + '/data.csv', 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['train loss'] + train_loss_list)  # 写list列表
    csv_writer.writerow(['test loss'] + test_loss_list)
    csv_writer.writerow(['train acc'] + train_acc_list)  # 写list列表
    csv_writer.writerow(['test acc'] + test_acc_list)
    f.close()


angle0 = {'p1': 1, 'pm': 1, 'pg': 1, 'cm': 1,
          'p2': 0, 'pmm': 0, 'pmg': 0, 'pgg': 0, 'cmm': 0,
          'p4': 0, 'p4m': 0, 'p4g': 0,
          'p3': 0, 'p3m1': 0, 'p31m': 0,
          'p6': 0, 'p6m': 0,
          'None': 1}
angle180 = {'p1': 0, 'pm': 0, 'pg': 0, 'cm': 0,
            'p2': 1, 'pmm': 1, 'pmg': 1, 'pgg': 1, 'cmm': 1,
            'p4': 1, 'p4m': 1, 'p4g': 1,
            'p3': 0, 'p3m1': 0, 'p31m': 0,
            'p6': 1, 'p6m': 1,
            'None': 0}
angle90 = {'p1': 0, 'pm': 0, 'pg': 0, 'cm': 0,
           'p2': 0, 'pmm': 0, 'pmg': 0, 'pgg': 0, 'cmm': 0,
           'p4': 1, 'p4m': 1, 'p4g': 1,
           'p3': 0, 'p3m1': 0, 'p31m': 0,
           'p6': 0, 'p6m': 0,
           'None': 0}
angle120 = {'p1': 0, 'pm': 0, 'pg': 0, 'cm': 0,
            'p2': 0, 'pmm': 0, 'pmg': 0, 'pgg': 0, 'cmm': 0,
            'p4': 0, 'p4m': 0, 'p4g': 0,
            'p3': 1, 'p3m1': 1, 'p31m': 1,
            'p6': 1, 'p6m': 1,
            'None': 0}

angle60 = {'p1': 0, 'pm': 0, 'pg': 0, 'cm': 0,
           'p2': 0, 'pmm': 0, 'pmg': 0, 'pgg': 0, 'cmm': 0,
           'p4': 0, 'p4m': 0, 'p4g': 0,
           'p3': 0, 'p3m1': 0, 'p31m': 0,
           'p6': 1, 'p6m': 1,
           'None': 0}

if __name__ == "__main__":
    # 训练参数
    img_size = 224
    data_num = 4000
    class_num = 2
    folder = 'D:/jupyter_data/images/minor-rotation'  # 文件即为folder/classes[idx]/*.png，分类：classes

    # 数据处理
    from torchvision.transforms.functional import crop


    def crop_left_upper(image):
        return crop(image, 0, 0, img_size, img_size)  # 裁剪左上角


    pipelines = {
        "train":
            transforms.Compose([
                # transforms.RandomRotation(degrees=15), # 随机旋转——变量1
                #             transforms.RandomHorizontalFlip(),
                transforms.Resize(img_size),  # 不改变长宽比——变量2
                transforms.RandomCrop(img_size),  # 随机裁剪——变量3
                #             transforms.Lambda(crop_left_upper)
            ]),
        "test":
            transforms.Compose([
                transforms.Resize(img_size),
                transforms.RandomCrop(img_size),  # RandomCrop->CenterCrop
            ])
    }

    class_dict = angle0
    model = CNN5(img_size, class_num)
    train_run(model, img_size, class_dict, folder, class_num, pipelines)
