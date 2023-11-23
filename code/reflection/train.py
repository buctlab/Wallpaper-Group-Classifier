import time
import os
import csv
import torch
from torchvision import transforms

from rotate.model_t import ModelT
from net.cnn import CNN5
from rotate.picture_loader import NoisePictureLoader17


def train_run(model, img_size, class_dict, folder, class_num, pipelines, data_num, epochs=20, batch_size=10, learning_rate=0.001):
    """
    running the model
    """
    data = time.strftime("%Y-%m-%d", time.localtime())
    present = time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime())
    save_path = 'output/{}/{}'.format(data, present)
    print('save_path:', save_path)

    model = ModelT(model=model, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, img_size=img_size,
                   classes=class_dict, pipelines=pipelines)
    # 数据
    train_load, test_load = model.data_load_17(class_num, folder, NoisePictureLoader17,
                                               data_num)  # NoisePictureLoader17:add noise

    if not os.path.exists(save_path):  # 如果不存在保存路径则创建
        os.makedirs(save_path)
    logger = model.get_logger(save_path + '/exp_{}_{}.log'.format(epochs, present))
    logger.info('整数周期, random crop, with noise, CNN')

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

    f = open(save_path + '/data.csv', 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['train loss'] + train_loss_list)  # 写list列表
    csv_writer.writerow(['test loss'] + test_loss_list)
    csv_writer.writerow(['train acc'] + train_acc_list)  # 写list列表
    csv_writer.writerow(['test acc'] + test_acc_list)
    f.close()


def reflection_run(folder, img_size=244, data_num=4000, class_num=4):
    """
    class_num = 4  # 分四类

    # 一次四分类
        |reflection | glide reflection|
        | --------- | ------------- |
        | 0 | 0 |
        | 0 | 1 |
        | 1 | 0 |
        | 1 | 1 |
    """
    class_dict = {'p1': 0, 'pm': 2, 'pg': 1, 'cm': 3,
                  'p2': 0, 'pmm': 2, 'pmg': 3, 'pgg': 1, 'cmm': 3,
                  'p4': 0, 'p4m': 2, 'p4g': 2,
                  'p3': 0, 'p3m1': 2, 'p31m': 2,
                  'p6': 0, 'p6m': 2,
                  'None': 0}
    pipelines = {
        "train":
            transforms.Compose([
                transforms.ToPILImage(),
                # transforms.RandomRotation(degrees=15), # 随机旋转——变量1
                # transforms.RandomHorizontalFlip(),
                transforms.Resize(img_size),  # 不改变长宽比——变量2
                transforms.CenterCrop(img_size),  # 随机裁剪——变量3
                #             transforms.Lambda(crop_left_upper)
                transforms.ToTensor(),
            ]),
        "test":
            transforms.Compose([
                transforms.ToPILImage(),
                # transforms.RandomHorizontalFlip(),
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),  # RandomCrop->CenterCrop
                transforms.ToTensor(),
            ])
    }
    model = CNN5(img_size, class_num)
    train_run(model, img_size, class_dict, folder, class_num, pipelines, data_num)


if __name__ == "__main__":
    folder = 'D:/jupyter_data/images/output'  # 文件即为folder/classes[idx]/*.png，分类：classes
    reflection_run(folder)
