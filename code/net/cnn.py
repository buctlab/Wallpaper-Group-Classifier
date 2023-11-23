import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from abc import ABC


def draw_features(width, height, x, save_name):
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width * height):
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)

        img = ((img - pmin) / (pmax - pmin + 0.0000001)) * 255
        img = img.astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        img = img[:, :, ::-1]

        plt.subplot(height, width, i + 1)
        plt.axis('off')
        plt.imshow(img)
    fig.savefig(save_name, dpi=100)
    fig.clf()
    plt.close()


class CNN5(nn.Module, ABC):
    def __init__(self, img_size, num_classes, is_save=False):
        super(CNN5, self).__init__()
        self.save = is_save
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 121, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(121),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(int((img_size / 32) ** 2 * 121), num_classes)
        # torch.nn.Dropout(0.5)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        if self.save:
            savepath = 'output/features_cat_heatmap'
            draw_features(4, 4, out2.cpu().numpy(), '{}/f2_conv.png'.format(savepath))
            draw_features(8, 8, out4.cpu().numpy(), '{}/f4_conv.png'.format(savepath))
            draw_features(11, 11, out5.cpu().numpy(), '{}/f5_conv.png'.format(savepath))

        out5 = out5.reshape(out5.size(0), -1)
        out5 = self.fc(out5)
        return out5