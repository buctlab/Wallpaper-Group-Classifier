'''
用于将小图形拼接成大图形
思路：只要计算出存放五种基本型的矩形贴在最终生成的图形上的左上角的点集即可
'''
import abc
import numpy as np
import math


class FullBase(abc.ABC):
    '''
        picture_width, picture_height：最终生成的图形的大小
        rectangle_width, rectangle_height：用于存放五种基本型的矩形的大小
        paste_points:点集，将存放五种基本型的矩形贴在最终生成的图形上的左上角的点集
    '''

    def __init__(self, picture_width, picture_height, rectangle_width, rectangle_height):
        self.picture_width = picture_width
        self.picture_height = picture_height
        self.rectangle_width = rectangle_width
        self.rectangle_height = rectangle_height
        self.paste_points = []

    @abc.abstractmethod
    def get_paste_points(self):
        pass


# 用于拼成最后大图的小画布是矩形/正方形
class RectangleFull(FullBase):
    '''
        用于拼成最后大图的小画布是矩形/正方形

        picture_width, picture_height：最终生成的图形的大小
        rectangle_width, rectangle_height：用于存放五种基本型的矩形的大小
    '''

    def __init__(self, picture_width, picture_height, rectangle_width, rectangle_height):
        super(RectangleFull, self).__init__(picture_width, picture_height, rectangle_width, rectangle_height)

    def get_paste_points(self):
        i = 0
        while i < self.picture_height:
            j = 0
            while j < self.picture_width:
                self.paste_points.append((j, i))
                j += self.rectangle_width
            i += self.rectangle_height

        return self.paste_points


# 用于拼成最后大图的小画布是平行四边形
class ObliqueFull(FullBase):
    '''
        用于拼成最后大图的小画布是平行四边形

        picture_width, picture_height：最终生成的图形的大小
        rectangle_width, rectangle_height：用于存放五种基本型的矩形的大小
        self.oblique_left:平行四边形左上角顶点坐标的x的值
        self.oblique_width：与小画布边重合的那条平行四边形的边的长度
        oblique_leftop_point：形如(x, y)的平行四边形的左上角的顶点坐标

    '''

    def __init__(self, picture_width, picture_height, rectangle_width, rectangle_height, oblique_leftop_point):
        super(ObliqueFull, self).__init__(picture_width, picture_height, rectangle_width, rectangle_height)
        self.oblique_left = oblique_leftop_point[0]
        self.oblique_width = self.rectangle_width - self.oblique_left

    def get_paste_points(self):
        i = -self.rectangle_height
        temp_y = 0  # 每一行的开始都需要向左移动来对上平行四边形的斜边
        while i < self.picture_height:
            j = temp_y
            while j < self.picture_width:
                self.paste_points.append((j, i))
                j += self.oblique_width
            i += self.rectangle_height
            temp_y += -self.oblique_left

        return self.paste_points

    def get_letfdown_theta(self, is_radian=True):
        '''
            获得平行四边形左下角的角度大小
        '''
        res = math.atan(self.rectangle_height / self.oblique_left)
        if is_radian:
            return res
        else:
            return res * 180 / math.pi


# 用于拼成最后大图的小画布是菱形
class RhombicFull(FullBase):
    '''
        用于拼成最后大图的小画布是菱形

        picture_width, picture_height：最终生成的图形的大小
        rectangle_width, rectangle_height：用于存放五种基本型的矩形的大小

        菱形的上顶点始终在rectangle_width的中点
        菱形的左顶点始终在rectangle_height的中点
    '''

    def __init__(self, picture_width, picture_height, rectangle_width, rectangle_height):
        super(RhombicFull, self).__init__(picture_width, picture_height, rectangle_width, rectangle_height)

    def get_paste_points(self):
        a = self.rectangle_width // 2
        b = self.rectangle_height // 2

        i = -self.rectangle_height
        while i < self.picture_height:
            j = -self.rectangle_width
            while j < self.picture_width:
                self.paste_points.append((j, i))
                self.paste_points.append((j + a, i + b))
                j += self.rectangle_width
            i += self.rectangle_height

        return self.paste_points

    def get_left_theta(self, is_radian=True):
        '''
            获得菱形左顶角的大小
        '''
        res = 2 * math.atan((self.rectangle_width / 2) / (self.rectangle_height / 2))
        if is_radian:
            return res
        else:
            return res * 180 / math.pi


# 用于拼成最后大图的小画布是正六边形
class HexagonalFull(FullBase):
    # 不给点的初始化
    def __init__(self, picture_width, picture_height, rectangle_width, rectangle_height, length):
        super(HexagonalFull, self).__init__(picture_width, picture_height, rectangle_width, rectangle_height)
        temp_h = int(length * np.sin(np.pi / 3))
        # temp_w = int(length * np.cos(np.pi / 3))
        temp_w = length // 2
        self.rightdown_shift_right = temp_w + length
        self.rightdown_shift_down = temp_h
        self.right_shift = temp_w * 2 + length + length

    # 以下是给三个点的方式的初始化
    '''
        用于拼成最后大图的小画布是正六边形

        picture_width, picture_height：最终生成的图形的大小
        rectangle_width, rectangle_height：用于存放五种基本型的矩形的大小
        hexpoint1：形如(x, y)的正六边形的右上角的顶点
        hexpoint2：形如(x, y)的正六边形的左上角的顶点
        hexpoint3：形如(x, y)的正六边形的最左边的顶点
    '''

    # def __init__(self, picture_width, picture_height, rectangle_width, rectangle_height, hexpoint1, hexpoint2, hexpoint3):
    #     super(HexagonalFull, self).__init__(picture_width, picture_height, rectangle_width, rectangle_height)        
    #     self.rightdown_shift_right = hexpoint1[0]
    #     self.rightdown_shift_down = hexpoint3[1]   
    #     self.right_shift = (hexpoint1[0] - hexpoint2[0]) + self.rectangle_width

    def get_paste_points(self):
        i = -self.rectangle_height
        while i < self.picture_height:
            j = -self.rectangle_width
            while j < self.picture_width:
                self.paste_points.append((j, i))
                self.paste_points.append((j + self.rightdown_shift_right, i + self.rightdown_shift_down))
                j += self.right_shift
            i += self.rectangle_height

        return self.paste_points
