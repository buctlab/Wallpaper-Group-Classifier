import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import numpy as np
import logging
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        # self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0


class ModelT:
    def __init__(self, **kwargs):  # epochs, batch_size, learning_rate, img_size, classes):
        # 定义超参数
        self.batch_size = kwargs.get('batch_size')
        self.learning_rate = kwargs.get('learning_rate')
        self.epochs = kwargs.get('epochs')
        self.device = kwargs.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # 数据集
        self.classes = kwargs.get('classes')
        self.train_list = []
        self.test_list = []

        self.img_size = kwargs.get('img_size', 224)
        self.model = kwargs.get('model').to(self.device)  # 分类
        # print(self.model)

        # Loss and optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1, last_epoch=-1)

        # 数据处理
        self.pipelines = kwargs.get('pipelines', {
            "train":
                transforms.Compose([
                    # transforms.RandomRotation(degrees=15), # 随机旋转
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize(self.img_size),  # 不改变长宽比
                    transforms.RandomCrop(self.img_size),  # 随机裁剪
                ]),
            "test":
                transforms.Compose([
                    transforms.Resize(self.img_size),
                    transforms.CenterCrop(self.img_size),  # RandomCrop->CenterCrop
                ])
        })
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])  # [0,1]变[-1,1]

    def get_logger(self, filename, verbosity=1, name=None):
        level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
        formatter = logging.Formatter("%(message)s")
        logger = logging.getLogger(name)  # 创建Logger
        logger.setLevel(level_dict[verbosity])

        fh = logging.FileHandler(filename, "w")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        logger.info(
            'classes:{} \n net:{} train:{} test:{} batch_size:{} learning_rate:{} device:{} img_size:{}'.format
            (self.classes, self.model.__class__.__name__, len(self.train_list), len(self.test_list), self.batch_size,
             self.learning_rate, self.device, self.img_size))

        return logger

    def getDictKey(self, myDict, value):
        # 取myDict字典中value对应的key
        return [k for k, v in myDict.items() if v == value]

    def data_load_17(self, class_num, folder, picture_loader, data_num=2000, data_load_kind='normal'):
        if data_load_kind == 'normal':
            for i in range(class_num):  # 分类
                list_name = []
                for classes in self.getDictKey(self.classes, i):  # 各类中种类不同
                    # print('walking_path:', os.path.join(folder, classes))
                    for root, _, _ in os.walk(os.path.join(folder, classes), topdown=False):
                        path = root + '/*.png'
                        list_name += [each for each in glob.glob(path)]
                    print(classes, '---->all: ', len(list_name))
                random.seed(0)
                random.shuffle(list_name)
                if len(list_name) > data_num > 0:
                    list_name = list_name[:data_num]  # 每一类只取data_num个数据，保证数据均一
                self.train_list += list_name[:int(len(list_name) * 0.7)]
                self.test_list += list_name[int(len(list_name) * 0.7):]
        elif data_load_kind == 'mix':
            for i in range(class_num):  # 分类
                list_name = []
                for classes in self.getDictKey(self.classes, i):  # 各类中种类不同
                    # print('walking_path:', os.path.join(folder, classes))
                    for root, _, _ in os.walk(os.path.join(folder[0], classes), topdown=False):
                        path = root + '/*.png'
                        list_name += [each for each in glob.glob(path)]
                    print(classes, '---->all: ', len(list_name))
                random.seed(0)
                random.shuffle(list_name)
                if len(list_name) > data_num > 0:
                    list_name = list_name[:data_num]  # 每一类只取data_num个数据，保证数据均一
                self.train_list += list_name[:int(len(list_name) * 0.7)]
                self.test_list += list_name[int(len(list_name) * 0.7):]

                list_name = []
                for classes in self.getDictKey(self.classes, i):  # 各类中种类不同
                    # print('walking_path:', os.path.join(folder, classes))
                    for root, _, _ in os.walk(os.path.join(folder[1], classes), topdown=False):
                        path = root + '/*.png'
                        list_name += [each for each in glob.glob(path)]
                    print(classes, '---->all: ', len(list_name))
                random.seed(0)
                random.shuffle(list_name)
                if len(list_name) > data_num > 0:
                    list_name = list_name[:data_num]  # 每一类只取data_num个数据，保证数据均一
                self.train_list += list_name[:int(len(list_name) * 0.7)]
                self.test_list += list_name[int(len(list_name) * 0.7):]
            data_load_kind = 'normal'
        else:
            for i in self.classes:  # 所有class分类
                for root, _, _ in os.walk(os.path.join(folder, i), topdown=False):
                    # path = self.folder_path + r'/{}/*.png'.format(i)
                    path = root + '/*.png'
                    # print('path:', path)
                    list_name = [each for each in glob.glob(path)]
                    if len(list_name) > data_num > 0:
                        list_name = list_name[:data_num]  # 每一类只取num个数据，保证数据均一
                    if list_name:
                        print(path, i, '---->', len(list_name))
                    self.train_list += list_name[:int(len(list_name) * 0.7)]
                    self.test_list += list_name[int(len(list_name) * 0.7):]

        train_set = picture_loader(list_name=self.train_list, classes=self.classes, norm=self.normalize,
                                   data_kind="train", pipelines=self.pipelines["train"], label_kind=data_load_kind)

        train_load = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

        # 解决DataLoader中一个Batch的图片需要尺寸相同的问题——修改collate_fn
        def collate_wrapper(batch):
            image_list, label_list = [], []
            # ----------------整数周期，先DFT再裁剪----------------
            img_size = 224
            complete_cycle_pipeline = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(img_size),  # 不改变长宽比——变量2
                transforms.CenterCrop(img_size),  # 随机裁剪——变量3
                transforms.ToTensor(),
            ])

            # --------------------------------------------
            for image, label in batch:
                image = complete_cycle_pipeline(image)
                image_list.append(image)
                # image_list.append(image.cpu().detach().numpy())
                label_list.append(label)
            # image_list = torch.Tensor(image_list)

            return torch.stack(image_list, dim=0), torch.Tensor(label_list)

        # train_load = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, collate_fn=collate_wrapper)

        test_set = picture_loader(list_name=self.test_list, classes=self.classes, norm=self.normalize,
                                  data_kind="test", pipelines=self.pipelines["test"], label_kind=data_load_kind)
        test_load = DataLoader(test_set, batch_size=self.batch_size, shuffle=True)
        print('train:', len(self.train_list), 'test:', len(self.test_list))
        return train_load, test_load

    def train_model(self, train_load):
        train_loss = 0
        loss_list = []
        self.model.train()
        correct_count = 0.0

        for i, (inputs, labels) in enumerate(train_load):
            self.optimizer.zero_grad()
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # forward
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            result = outputs.argmax(dim=1)
            # print('result:', result)
            correct_count += result.eq(labels.view_as(result)).sum().item()

            # backward and optimize
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            loss_list.append(loss.item())
            if i % 50 == 0:
                print("Train Loss: {}".format(loss.item()))
        train_acc = correct_count / len(train_load.dataset)
        self.scheduler.step()
        return loss_list, train_loss / len(train_load), train_acc

    def test_model(self, test_load):
        test_loss = 0
        self.model.eval()
        correct_count = 0.0
        with torch.no_grad():
            for inputs, labels in test_load:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                prediction = outputs.argmax(dim=1)
                correct_count += prediction.eq(labels.view_as(prediction)).sum().item()

                test_loss += loss.item()
            accuracy = correct_count / len(test_load.dataset)
            # test_acc_list.append(accuracy)
            print("Accuracy: {:.6f}".format(accuracy))
        return test_loss / len(test_load), accuracy

    def process_figure(self, train_loss, test_loss, train_acc, test_acc, epochs, save_path, present):
        x_lines = len(test_acc)
        x = np.linspace(0, x_lines, x_lines)
        x_t = np.linspace(0, x_lines, len(train_loss))
        print(len(x), len(x_t))
        with plt.style.context(['science', 'grid']):
            fig, ax = plt.subplots()
            ax.plot(x_t, train_loss, label='train loss')
            ax.plot(x, test_loss, label='test loss')
            ax.plot(x, train_acc, label='train acc')
            ax.plot(x, test_acc, label='test acc')
            ax.legend(title='Order', framealpha=0.85)
            ax.set(xlabel='Epoch')
            ax.autoscale(tight=True)

        plt.savefig(save_path + '/loss_acc_epoch{}_{}.png'.format(epochs, present), dpi=700)
