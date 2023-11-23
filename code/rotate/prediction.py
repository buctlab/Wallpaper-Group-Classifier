import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import glob
import os
import csv

from net.cnn import CNN5
from rotate.picture_loader import NoisePictureLoader17
from rotate.plot_figure import plot_CM


def confusion_matrix(preds, labels, conf_matrix):
    # update confusion matrix
    for p, t in zip(preds, labels):
        conf_matrix[t, p] += 1
    return conf_matrix


class Prediction:
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs:
            model: 模型结构
            device:
            batch_size:
            img_size: 训练图片大小
            data_num: 参与训练的数据个数

            classes: list: [class_label_name]分类类别
            class_num: classes分类个数

            pth_file: 训练好的模型数据路径
            folder_path: 测试数据路径
            cm_save_path:存储cm矩阵
            title: cm矩阵titles
        """
        self.model = kwargs.get('model')
        self.device = kwargs.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = kwargs.get('batch_size', 20)
        self.img_size = kwargs.get('img_size', 224)
        self.data_num = kwargs.get('num', -1)

        self.classes = kwargs.get('classes')
        self.class_num = kwargs.get('class_num', len(self.classes))

        self.pth_file = kwargs.get('pth_file')
        self.folder_path = kwargs.get('folder_path')
        self.cm_save_path = os.path.split(self.pth_file)[0]
        print('cm_save_path:', self.cm_save_path)
        # self.folder_classes = kwargs.get('folder_classes', self.classes)
        self.title = kwargs.get('title')

        # 网络加载
        self.model.load_state_dict(torch.load(self.pth_file))
        self.model = self.model.to(self.device)

    def pre_model(self, load, class_dict):
        criterion = nn.CrossEntropyLoss()
        pre_loss = 0
        self.model.eval()
        correct_count = 0.0

        # 创建一个空矩阵存储预测矩阵
        matrix = torch.zeros(len(class_dict), self.class_num)  # 存储17类中每一类预测结果

        pre = []
        # group = class_dict.keys()
        # group_value = class_dict.values()
        with torch.no_grad():
            for inputs, group_labels, labels in load:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)

                # print('output:', outputs)
                # print('label:', group_labels, labels)

                loss = criterion(outputs, labels)

                prediction = outputs.argmax(dim=1)

                # print('pre:', prediction)
                pre += prediction.cpu().numpy().tolist()
                correct_count += prediction.eq(labels.view_as(prediction)).sum().item()
                # 预测矩阵
                matrix = confusion_matrix(prediction, labels=group_labels, conf_matrix=matrix)

                pre_loss += loss.item()
            accuracy = correct_count / len(load.dataset)
            print('acc/data_num', correct_count, len(load.dataset))
            print("Accuracy: {:.6f}".format(accuracy))
            print('loss: {:.6f}'.format(pre_loss / len(load.dataset)))

        if self.is_show:
            plot_CM(matrix.numpy(), x_ticks=self.classes, y_ticks=class_dict.keys(), title=self.title,
                    save_path=self.cm_save_path + '_pre_F(acc{:.2f}_[{:.0f}_{}]).png'.format(accuracy, correct_count,
                                                                                             len(load.dataset)))
            plot_CM(matrix.numpy(), x_ticks=self.classes, y_ticks=class_dict.keys(), normalize=True, title=self.title,
                    save_path=self.cm_save_path + '_pre_T(acc{:.2f}_[{:.0f}_{}]).png'.format(accuracy, correct_count,
                                                                                             len(load.dataset)))
            # 混淆矩阵
            conf_matrix = torch.zeros(self.class_num, self.class_num)
            i = 0
            for value in class_dict.values():
                conf_matrix[value] += matrix[i]
                i += 1

            # 数据写入文件
            f = open(self.cm_save_path + '(acc{:.2%}_[{:.0f}_{}])_pre_data.csv'.format(accuracy, correct_count,
                                                                                       len(load.dataset)), 'w',
                     newline='', encoding='utf-8')
            csv_writer = csv.writer(f)
            csv_writer.writerow(self.classes)
            for data in matrix.numpy():
                csv_writer.writerow(data)  # 写list列表
            f.close()

            plot_CM(conf_matrix.numpy(), x_ticks=self.classes, y_ticks=self.classes, normalize=False,
                    title=self.title,
                    save_path=self.cm_save_path + '_cm_F(acc{:.2f}_[{:.0f}_{}]).png'.format(accuracy, correct_count,
                                                                                            len(load.dataset)))
            plot_CM(conf_matrix.numpy(), x_ticks=self.classes, y_ticks=self.classes, normalize=True,
                    title=self.title,
                    save_path=self.cm_save_path + '_cm_T(acc{:.2f}_[{:.0f}_{}]).png'.format(accuracy, correct_count,
                                                                                            len(load.dataset)))
        return pre

    def getDictKey(self, myDict, value):
        return [k for k, v in myDict.items() if v == value]

    def prediction_run17(self, picture_loader, class_dict, data_load_kind='normal', label_kind='group', is_show=False):
        '''

        Args:
            picture_loader:
            class_dict: map:{group_label:class_label}
            data_load_kind: 测试数据的获取方式，'normal': 当前分类中每类取均一数据，else:17类中每类都取均一数据
            is_show: prediction result

        Returns: self.pre_model(pre_load, class_dict)

        '''
        self.is_show = is_show
        self.fig_name = '{}_{}_{}'.format(self.data_num, self.classes, tuple(class_dict))
        pipelines = {
            "test":
                transforms.Compose([
                    transforms.Resize(self.img_size),
                    transforms.CenterCrop(self.img_size)
                ])}
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])  # [0,1]变[-1,1]

        # 17类数据
        pre_list = []
        if data_load_kind == 'normal':
            for i in range(self.class_num):  # 按class分类
                list_name = []
                for classes in self.getDictKey(class_dict, i):  # 各类中种类不同
                    for root, _, _ in os.walk(os.path.join(self.folder_path, classes), topdown=False):
                        path = root + '/*.png'
                        # print('path:', path)
                        list_name += [each for each in glob.glob(path)]
                # random.shuffle(list_name)
                if len(list_name) > self.data_num > 0:
                    list_name = list_name[:self.data_num]  # 每一类只取num个数据，保证数据均一
                print(self.classes[i], '---->', len(list_name))
                pre_list += list_name
        elif data_load_kind == 'lattice':
            for i in class_dict:  # 所有class分类
                for root, _, _ in os.walk(os.path.join(self.folder_path, i), topdown=False):
                    # path = self.folder_path + r'/{}/*.png'.format(i)
                    path = root + '/*.png'
                    # print('path:', path)
                    list_name = [each for each in glob.glob(path)]
                    if len(list_name) > self.data_num > 0:
                        list_name = list_name[:self.data_num]  # 每一类只取num个数据，保证数据均一
                    if list_name:
                        print(i, '---->', len(list_name))
                    pre_list += list_name
        else:
            for i in class_dict.keys():  # 所有class分类
                for root, _, _ in os.walk(os.path.join(self.folder_path[0], i), topdown=False):
                    # path = self.folder_path + r'/{}/*.png'.format(i)
                    path = root + '/*.png'
                    # print('path:', path)
                    list_name = [each for each in glob.glob(path)]
                    if self.data_num < len(list_name):
                        list_name = list_name[:self.data_num]  # 每一类只取num个数据，保证数据均一
                    if list_name:
                        print(i, '---->', len(list_name))
                    pre_list += list_name
                for root, _, _ in os.walk(os.path.join(self.folder_path[1], i), topdown=False):
                    # path = self.folder_path + r'/{}/*.png'.format(i)
                    path = root + '/*.png'
                    # print('path:', path)
                    list_name = [each for each in glob.glob(path)]
                    if self.data_num < len(list_name):
                        list_name = list_name[:self.data_num]  # 每一类只取num个数据，保证数据均一
                    if list_name:
                        print(i, '---->', len(list_name))
                    pre_list += list_name

        pre_set = picture_loader(list_name=pre_list, classes=class_dict, norm=normalize,
                                 data_kind="test", label_kind=label_kind,  # 'normal','lattice','group'
                                 pipelines=pipelines["test"])  # test文件夹图片输出
        pre_load = DataLoader(pre_set, batch_size=self.batch_size, shuffle=False)
        return self.pre_model(pre_load, class_dict)


if __name__ == '__main__':
    # 二分类——angle 0
    pthfile = 'output/info/2021-5-14/2021-05-14_22.47.10/epoch_18_20_2021-05-14_22.47.10.pkl'
    classes = ['else', 'angle 0']  # 分类类别名称
    # 分类文件名：类别号
    class_dict = {'p1': 1, 'pm': 1, 'pg': 1, 'cm': 1,
                  'p2': 0, 'pmm': 0, 'pmg': 0, 'pgg': 0, 'cmm': 0,
                  'p4': 0, 'p4m': 0, 'p4g': 0,
                  'p3': 0, 'p3m1': 0, 'p31m': 0,
                  'p6': 0, 'p6m': 0,
                  'None': 1, }
    img_size = 224  # 图片大小
    num = 2  # 数据个数
    folder_path = r'D:/jupyter_data/images/test'

    net_save = CNN5(img_size, len(classes), is_save=True)
    pre = Prediction(pth_file=pthfile, model=net_save, img_size=img_size, classes=classes, batch_size=10,
                     title='Rotation Angle $0^\circ$', folder_path=folder_path, num=num).prediction_run17(
        NoisePictureLoader17, class_dict, data_load_kind='normal', is_show=True)
