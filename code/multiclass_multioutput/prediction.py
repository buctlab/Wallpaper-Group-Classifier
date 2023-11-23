import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import glob
import os

from net.cnn import CNN5
from multiclass_multioutput.picture_loader import MultiLoader

import pandas as pd


# 更新混淆矩阵
def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


class Prediction:
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs:
            class_num: 分类个数
            class_num: 参与训练的类个数
            per_class_num: 参与训练的每一类中数据个数
        """
        self.img_size = kwargs.get('img_size', 224)
        self.classes = kwargs.get('classes')

        self.folder_path = kwargs.get('folder_path')
        self.device = kwargs.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = kwargs.get('batch_size', 1)

        self.class_num = kwargs.get('class_num', len(self.classes))
        self.per_class_num = kwargs.get('per_class_num', -1)

    def val_model(self, load, model):
        model.eval()

        differ = []
        m = torch.nn.Softmax(dim=1)
        with torch.no_grad():
            for inputs, labels in load:
                inputs = inputs.to(self.device)
                outputs = model(inputs)

                temp = m(outputs).cpu().numpy()
                if len(temp[0]) == 2:
                    temp = temp[:, 1]
                differ += temp.tolist()

        return differ

    def getDictKey(self, myDict, value):
        return [k for k, v in myDict.items() if v == value]

    def data17(self, loader, is_show=False, data_load_kind='normal'):
        """
        Args:
            classes_name: ['name1','name2','name3']
            is_show:

        Returns: val_load

        """
        self.is_show = is_show

        pipelines = {
            "valid":
                transforms.Compose([
                    transforms.ToPILImage(),
                    # transforms.RandomHorizontalFlip(),
                    transforms.Resize(self.img_size),
                    transforms.CenterCrop(self.img_size),  # RandomCrop->CenterCrop
                    transforms.ToTensor(),
                ])}
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])  # [0,1]变[-1,1]

        # 数据
        val_list = []

        data_class_name = []

        class_num = 0
        if data_load_kind == 'normal':
            for classes_list in self.classes:
                print(classes_list)
                for classes in classes_list:
                    if class_num < self.class_num:
                        class_num += 1
                        list_name = []
                        for root, _, _ in os.walk(os.path.join(self.folder_path, classes), topdown=False):
                            # print('root:', root)
                            path = root + '/*.png'
                            list_name += [each for each in glob.glob(path)]
                        if self.per_class_num < len(list_name):
                            list_name = list_name[:self.per_class_num]  # 每一类只取num个数据，保证数据均一
                        print(classes, '---->', len(list_name))
                        val_list += list_name
                        data_class_name += [classes] * len(list_name)

        elif data_load_kind == 'single':
            i = 0
            for root, _, _ in os.walk(self.folder_path, topdown=False):
                # print('root:', root)
                path = root + '/*.jpg'
                val_list += [each for each in glob.glob(path)]
                print(path, '---->', len(val_list))
                data_class_name += str(i) * len(val_list)
                i += 1
            print('val_list:', val_list)
            pd.DataFrame(val_list).to_csv('E:/Desktop/conf_matrix/IGP_result.csv')

        print('total:', len(val_list))
        val_set = loader(list_name=val_list, classes=self.classes, norm=normalize,
                         pipelines=pipelines["valid"])  # valid文件夹图片输出
        val_load = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        return val_load, data_class_name

    def prediction_run17(self, model, val_load, pth_file):
        # 网络加载
        model.load_state_dict(torch.load(pth_file))
        model = model.to(self.device)
        return self.val_model(val_load, model)


if __name__ == '__main__':
    pthfile = '../rotate/output/info/pkl/direct_pkl/1-epoch_16_20_2021-01-12_14.20.18.pkl'
    classes_name = [['p1', 'pm', 'pg', 'cm'],
                    ['p2', 'pmm', 'pmg', 'pgg', 'cmm'],
                    ['p4', 'p4m', 'p4g'],
                    ['p3', 'p3m1', 'p31m'],
                    ['p6', 'p6m']]

    img_size = 224
    folder_path = 'D:/result'

    base_classifier = CNN5(img_size, 2)

    prediction = Prediction(img_size=img_size, classes=classes_name, folder_path=folder_path,
                            class_num=17,
                            per_class_num=3)
    data_load, data_class_name = prediction.data17(MultiLoader)
    pre_result = prediction.prediction_run17(base_classifier, data_load, pth_file=pthfile)
    print(pre_result)
