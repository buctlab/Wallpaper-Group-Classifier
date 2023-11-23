# coding=utf-8
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import copy
import os
from base.overspread import *


def rotatedRectWithMaxArea(w, h, angle):
    """
    去除旋转中出现的图像黑边
    """
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    sin_a, cos_a = abs(np.sin(angle)), abs(np.cos(angle))
    if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a
    return wr, hr


class Cell:
    def __init__(self, **kwargs):
        """
        :param
        picture_width,picture_height,GroundImg : canvas
        rectangle_width,rectangle_height, markImg : lattie
        point : cut point (ndarray)
        source_image_path : material
        save_image_path: save GroundImg
        img : source_image_path image
        kind : cell
        info: point info
        is_save:
        is_show:
        """
        self.picture_width = kwargs.pop('picture_width', 600)
        self.picture_height = kwargs.pop('picture_height', 600)
        self.GroundImg = Image.new('RGBA', (self.picture_width, self.picture_height), 0)

        self.rectangle_width = kwargs.pop('rectangle_width', 200)
        self.rectangle_height = kwargs.pop('rectangle_height', 100)
        self.markImg = Image.new('RGBA', (self.rectangle_width, self.rectangle_height), 0)  # diaphaneity = 0 or 'red'

        self.kind = kwargs.get('kind')
        self.info = kwargs.get('info')

        self.source_image_path = kwargs.pop('source_image_path', "../images/dataset/1.png")
        self.point = kwargs.pop('point')
        self.img = Image.open(self.source_image_path).convert("RGBA").resize(np.max(self.point, axis=0))

        self.angle = kwargs.pop('angle', 0)
        print('angle:', self.angle)
        print('info:', self.info)

        self.is_save = kwargs.pop('is_save', True)
        if 'save_image_path' in kwargs:  # 传入保存路径才保存
            self.save_image_path = kwargs.pop('save_image_path')
        else:
            self.is_save = False
        self.is_show = kwargs.pop('is_show', False)

        self.cut()

    def rotate(self, angle, pos, center=None, expand=True):
        # 旋转 Image.NEAREST
        if center is None:
            img = self.img.rotate(angle, Image.NEAREST, expand=expand)
        else:
            img = self.img.rotate(angle, Image.NEAREST, center=center, expand=expand)
        self.paste_img(pos, img)

    def rotate_mark(self, angle, center=None, expand=True):
        # 旋转
        if center is None:
            self.markImg = self.markImg.rotate(angle, Image.NEAREST, expand=expand)
        else:
            self.markImg = self.markImg.rotate(angle, Image.NEAREST, center=center, expand=expand)

    def tb_mirror(self, pos):
        # 上下镜像
        img = self.img.transpose(Image.FLIP_TOP_BOTTOM)
        self.paste_img(pos, img)

    def lr_mirror(self, pos):
        # 左右镜像
        img = self.img.transpose(Image.FLIP_LEFT_RIGHT)
        self.paste_img(pos, img)

    def mirror_angle(self, angle, pos, center=None, expand=True):
        # 斜镜像
        img = self.img.transpose(Image.FLIP_LEFT_RIGHT)  # lr mirror
        if center is None:
            img = img.rotate(angle, expand=expand)
        else:
            img = img.rotate(angle, center=center, expand=expand)
        self.paste_img(pos, img)

    def paste_img(self, pos, img=None):
        if not img:
            img = self.img
        _, _, _, mask = img.split()
        self.markImg.paste(img, pos, mask=mask)

    def cut(self):
        (x1, y1) = np.max(self.point, axis=0)  # img cut area
        # cut
        self.img = self.img.crop((0, 0, x1, y1))
        # print(self.point)
        point = copy.deepcopy(self.point)
        point[:, 1] = self.point[:, 0] - 0
        point[:, 0] = self.point[:, 1] - 0

        alpha = np.zeros(self.img.size, dtype="uint8")  # initialization

        roi_t = np.expand_dims(point, axis=0)

        cv2.fillPoly(alpha, roi_t, 255)  # opaque fill

        alpha = alpha.transpose()
        alpha = Image.fromarray(alpha)
        self.img.putalpha(alpha)
        # return alpha

    def show(self):
        if self.is_show:
            plt.subplot(131), plt.imshow(self.img)
            plt.subplot(132), plt.imshow(self.markImg)
            plt.subplot(133), plt.imshow(self.GroundImg)
            plt.show()

    def border(self):
        roi_t = np.expand_dims(self.point, axis=0)
        _, _, _, alpha = self.img.split()
        img = np.array(self.img)
        cv2.polylines(img, roi_t, 1, (190, 190, 190))  # border
        self.img = Image.fromarray(img)
        self.img.putalpha(alpha)

    def update_img(self):
        # markImg -> img
        self.img = copy.deepcopy(self.markImg)

    def over_spread(self, oblique_point=None):
        # 铺满大画布
        if self.kind == 1 or self.kind == 2:  # 矩形/正方形
            l = RectangleFull(self.picture_width, self.picture_height, self.rectangle_width, self.rectangle_height)
            points = l.get_paste_points()

        elif self.kind == 3:  # 平行四边形
            l = ObliqueFull(self.picture_width, self.picture_height, self.rectangle_width, self.rectangle_height,
                            oblique_point)
            points = l.get_paste_points()

        elif self.kind == 4:  # 菱形
            l = RhombicFull(self.picture_width, self.picture_height, self.rectangle_width, self.rectangle_height)
            points = l.get_paste_points()

        elif self.kind == 5:  # 六边形
            l = HexagonalFull(self.picture_width, self.picture_height, self.rectangle_width, self.rectangle_height,
                              self.info[0])  # 传六边形边长
            points = l.get_paste_points()
        else:
            raise ValueError("kind should between 1 and 5")
        _, _, _, mask = self.markImg.split()
        for i in range(len(points)):
            self.GroundImg.paste(self.markImg, points[i], mask=mask)

    def ground_img_rotate(self):
        if self.angle:
            w_ori, h_ori = self.GroundImg.size
            wr, hr = rotatedRectWithMaxArea(w_ori, h_ori, np.radians(self.angle))
            img = self.GroundImg.rotate(self.angle, Image.NEAREST, expand=True)
            w_rot, h_rot = img.size
            self.GroundImg = img.crop(
                ((w_rot - wr) / 2, (h_rot - hr) / 2, (w_rot - wr) / 2 + wr, (h_rot - hr) / 2 + hr))

    def save(self):
        if self.is_save:
            img_name, _ = os.path.splitext(os.path.basename(self.source_image_path))
            folder_path = os.path.join(self.save_image_path, self.__class__.__name__, str(self.kind))
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            path = os.path.join(folder_path,
                                '{i}_{j}_{k}_angle{l}.png'.format(i=img_name, j=self.kind, k=self.info, l=self.angle))
            print('img save path:', path)
            self.GroundImg.save(path)

    def run(self, func_list):
        """
        run function list.
        Args:
            func_list: (function, parameter)
                    such as 'paste_img','update_img','rotate','over_spread','show'.
        """
        for func, *parameter in func_list:
            eval('self.' + func)(*parameter)
        self.ground_img_rotate()
