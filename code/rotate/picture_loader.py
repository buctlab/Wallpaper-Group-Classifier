import os
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import cv2
import math

np.random.seed(0)


def sp_noise(img, num=4000):
    new_img = np.copy(img)
    for i in range(num):  # 添加点噪声
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        sp = np.random.randint(0, 2)
        new_img[temp_x][temp_y] = sp * 255  # 在temp_x,temp_y处添加椒盐噪声（255是白色，0是黑色）
    return new_img


def gaussian_noise(img, mean=0.0, var=0.001):  # 均值，方差
    img = np.array(img / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    img2 = img + noise
    if img2.min() < 0:
        clip = -1
    else:
        clip = 0
    img3 = np.clip(img2, clip, 1.0) * 255
    return img3


def choose_windows(name='Hanning', N=20):
    # Rect/Hanning/Hamming
    if name == 'Hamming':
        window = np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
    elif name == 'Hanning':
        window = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
    elif name == 'Rect':
        window = np.ones(N)
    elif name == 'Gaussian':
        alpha = 3
        window = np.array([math.exp(-1 / 2 * (alpha * 2 * (n - N / 2) / (N - 1)) ** 2) for n in range(N)])
    else:
        raise ValueError('unknown window_name')
    return window


def img_add_window(image, window_name='Hanning'):
    # window
    Window = choose_windows(window_name, N=np.size(image, 1)) * choose_windows('Hanning', N=np.size(image, 0)).reshape(
        -1, 1)
    # print('window & image size:', Window.shape, image.shape)
    return image * Window


def filter(img, name):
    if name == 'Gaussian':
        return cv2.GaussianBlur(img, (5, 5), 1)  # 高斯
    elif name == 'Bilateral':
        return cv2.bilateralFilter(img, 9, 100, 100)  # 双边
    else:
        raise ValueError('unknown filter_name')


def fft(img):
    f = np.fft.fft2(img)  # 二维的傅里叶变换(img,n=NOne,axis=-1,norm=None)
    fshift = np.fft.fftshift(f)  # 中心化(将FFT输出中的直流分量移动到频谱中央)

    s = np.log(np.abs(fshift) / fshift.size ** 0.5 + 1)  # （/ fshift.size ** 0.5）使能量前后相等
    # s = np.log(np.abs(fshift) + 1)  # ndarray
    s_new = (s - np.min(s)) / (np.max(s) - np.min(s))  # 归一化到【0,1】

    t_s = torch.tensor(s_new, dtype=torch.float32)  # tensor
    img_fft = t_s.unsqueeze(0)  # (W,H)->(C,H,W)
    return img_fft


class NoisePictureLoader17(data.Dataset):
    r"""pictures loader, the pictures' are arranged in this way:

        root/p1/*/1_xxx.jpg
        root/p2/*/1_xxx.jpg
        ...
        root/p3/*/13_xxx.jpg

    e.g.
    classes = {'p1': 0,'pm':0,'pg':0,'cm':0,
              'p2':1,'pmm':1,'pmg':1,'pgg':1,'cmm':1,
              'p4': 2, 'p4g': 2, 'p4g':2,
              'p3': 3,'p3m1':3,'p31m':3,
              'p6': 4,'p6m':4}
    """

    def __init__(self, list_name, norm, classes, data_kind=None, pipelines=None, **kwargs):
        """
        Args:
            classes: (map:'class','group')(list:'lattice')
            group_classes: (map) get from classes, 用于label_kind='group'
            data_kind: 'train','test'
            label_kind: 获取标签方式: 'normal','lattice','group'

            pipelines: pipeline['train','test']
            list_name: list: figure path
            normalize: [0,1]->[-1,1]

            filter_name:
            window_name:
            **kwargs:
        """
        self.classes = classes
        self.data_kind = data_kind
        self.label_kind = kwargs.pop('label_kind', 'normal')
        if self.label_kind == 'group':
            class_keys = self.classes.keys()
            self.group_classes = dict(zip(list(class_keys), range(len(class_keys))))

        self.pipelines = pipelines
        self.list_name = list_name
        self.normalize = norm

        self.filter_name = kwargs.pop('filter_name', 'Gaussian')
        self.window_name = kwargs.pop('window_name', 'Hanning')
        # self.data_preprocess_kind = kwargs.pop('data_preprocess_kind', None)

    def __getitem__(self, index):
        img = Image.open(self.list_name[index]).convert('L')
        # print(self.list_name[index])

        # -----------------图像处理开始-------------------
        if self.pipelines:
            img = self.pipelines(img)

        image = np.array(img)
        img = 255 * ((image - np.min(image)) / (np.max(image) - np.min(image)))  # (0~255)
        if self.data_kind == 'train':
            # 随机添加噪声
            temp = np.random.randint(0, 2)
            if temp == 0:
                img = sp_noise(image)
            else:
                img = gaussian_noise(image)

        # if self.data_preprocess_kind == 'all':
        # 加窗
        window_img = img_add_window(img, window_name=self.window_name)
        # 滤波
        filter_img = filter(window_img, name=self.filter_name)
        # 频谱变换
        fft_img = fft(filter_img)  # 变换到频域上(return tensor fft:[0,1])
        img = fft_img
        # else:
        # img = torch.tensor(filter_img, dtype=torch.float32)
        # img = img.unsqueeze(0)
        # 归一化
        norm_img = self.normalize(img)  # 归一化[0,1]变[-1,1]
        # -----------------图像处理结束-------------------

        # if self.pipelines:
        #     norm_img = self.pipelines(norm_img)

        # ----------------标签：在名字上------------------
        if self.label_kind == 'normal':
            # 普通n分类
            # 取分类类别数, 0 ~ n
            group = self.list_name[index].split('\\')[-3]
            label = self.classes[group]
            # print('label:', label)
            return norm_img, label
        elif self.label_kind == 'real':
            group = self.list_name[index].split('\\')[-3]
            # print('group:', group, self.classes)
            label = self.classes.get(group)
            if label == None:
                label = 2
            # print('label:', label)
            return norm_img, label
        elif self.label_kind == 'lattice':
            # basic_shape: 五分类
            label = int(self.list_name[index].split('\\')[-2]) - 1
            # return norm_img, label, label # ???
        elif self.label_kind == 'group':
            # 用于预测
            # group:p1,pmm,pmg,p2,p3...
            group = self.list_name[index].split('\\')[-3]
            # print(self.list_name[index], group)
            group_label = self.group_classes[group]
            label = self.classes[group]
            return norm_img, group_label, label
        else:
            raise ValueError('unknown label_kind')
        # ------------------------------------------------

    def __len__(self):
        return len(self.list_name)


if __name__ == '__main__':
    import random

    source_folder = r'E:\jupyter_tree\group\结果汇总\噪声、随机裁剪、旋转的代表性图片\随机裁剪-旋转'
    files = os.listdir(source_folder)
    # files.sort(key=lambda x: int(x[:-4]))
    random.shuffle(files)
    for filename in files:
        source_image_path = os.path.join(source_folder, filename)

        img = Image.open(source_image_path).convert('L')
        image = np.array(img)
        img = 255 * ((image - np.min(image)) / (np.max(image) - np.min(image)))  # (0~255)

        # 随机添加噪声
        temp = np.random.randint(0, 2)
        if temp == 0:
            img = sp_noise(image)
        else:
            img = gaussian_noise(image)

        # img = torch.tensor(img, dtype=torch.float32)
        # img = img.unsqueeze(0)
        # 归一化
        # normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        # norm_img = normalize(img)

        Image.fromarray(np.uint8(img)).save('E:\jupyter_tree\group\结果汇总\噪声、随机裁剪、旋转的代表性图片\噪声-随即裁剪/' + filename)

        # norm_img.save('E:\jupyter_tree\group\结果汇总\噪声、随机裁剪、旋转的代表性图片\噪声-随即裁剪/'+filename)
        # 频谱变换
        # fft_img = fft(filter_img)  # 变换到频域上(return tensor fft:[0,1])
        # img = fft_img
