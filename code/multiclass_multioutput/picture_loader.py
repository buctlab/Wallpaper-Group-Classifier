import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
import cv2
import math


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


class MultiLoader(data.Dataset):
    r"""
    pictures loader, the pictures are arranged in this way:

        root/p1/*/1_xxx.jpg
        root/p2/*/1_xxx.jpg
        ...
        root/p3/*/13_xxx.jpg

    Args:
        root(string) : like the info above
        data_kind(string) : 区分训练集/测试集/验证集
        root/p3/*/xxx.jpg : p3 is the label

    e.g.
    classes = {'p1','pm','pg','cm',
              'p2','pmm','pmg','pgg','cmm',
              'p4', 'p4g', 'p4g',
              'p3','p3m1','p31m',
              'p6','p6m'}
    """

    def __init__(self, list_name, norm, classes, pipelines=None, **kwargs):
        self.pipelines = pipelines
        self.list_name = list_name
        self.normalize = norm
        self.classes = classes

        self.filter_name = kwargs.pop('filter_name', 'Gaussian')
        self.window_name = kwargs.pop('window_name', 'Hanning')

    def __getitem__(self, index):
        img = Image.open(self.list_name[index]).convert('L')
        image = np.array(img)
        img = 255 * ((image - np.min(image)) / (np.max(image) - np.min(image)))  # (0~255)
        # 加窗
        window_img = img_add_window(img, window_name=self.window_name)
        # 滤波
        filter_img = filter(window_img, name=self.filter_name)
        # 频谱变换
        fft_img = fft(filter_img)  # 变换到频域上(return tensor fft:[0,1])
        # 归一化
        norm_img = self.normalize(fft_img)  # 归一化[0,1]变[-1,1]

        if self.pipelines:
            norm_img = self.pipelines(norm_img)

        # img2 = self.gaussian(img)
        # img3 = self.fft(img2)  # 变换到频域上
        label = self.list_name[index].split('\\')[-3]

        return norm_img, label

    def __len__(self):
        return len(self.list_name)


class MirrorLoader(data.Dataset):
    """
    pictures loader, the pictures' are arranged in this way:

        root/p1/rectangle/train/1_xxx.jpg
        root/p1/rectangle/test/1_xxx.jpg
        root/p1/rectangle/valid/1_xxx.jpg
        ...
        root/p3/13_xxx.jpg

    Args:
        root(string):like the info above
        data_kind(string):区分训练集/测试集/验证集
        1_xxx.jpg:1 is the label

    class_name = [
           "cm", "cmm", "p1", "p2", "p3",
           "p3m1", "p4", "p4g","p4m", "p6",
           "p6m", "p31m", "pg", "pgg", "pm",
           "pmg", "pmm"]
    """

    def __init__(self, list_name, norm, classes, pipelines=None, **kwargs):
        self.pipelines = pipelines
        self.list_name = list_name
        self.normalize = norm
        self.classes = classes

        self.filter_name = kwargs.pop('filter_name', 'Gaussian')
        self.window_name = kwargs.pop('window_name', 'Hanning')

    def __getitem__(self, index):
        img = Image.open(self.list_name[index]).convert('L')

        image = np.array(img)
        img = 255 * ((image - np.min(image)) / (np.max(image) - np.min(image)))  # (0~255)
        # 加窗
        window_img = img_add_window(img, window_name=self.window_name)
        # 滤波
        filter_img = filter(window_img, name=self.filter_name)
        # 频谱变换
        fft_img = fft(filter_img)  # 变换到频域上(return tensor fft:[0,1])
        # 归一化
        norm_img = self.normalize(fft_img)  # 归一化[0,1]变[-1,1]

        if self.pipelines:
            norm_img = self.pipelines(norm_img)

        label = self.list_name[index].split('\\')[-3]

        return norm_img, label

    def __len__(self):
        return len(self.list_name)