import os
from base.seventeen_wallpaper import *
from base.point import Point
import numpy as np
import random
from PIL import Image
from base.cell import rotatedRectWithMaxArea


def produce_None(source_folder, save_image_path, img_num, angle):  # 第18类
    files = os.listdir(source_folder)
    random.shuffle(files)
    for filename in files[0:img_num]:  # os.listdir(source_folder):
        source_image_path = os.path.join(source_folder, filename)

        img = Image.open(source_image_path)
        w_ori, h_ori = img.size
        wr, hr = rotatedRectWithMaxArea(w_ori, h_ori, np.radians(angle))  # 计算保留区域大小

        result_img = img.rotate(angle, Image.NEAREST, expand=True)  # 旋转
        w_rot, h_rot = result_img.size

        rotated_img = result_img.crop(
            ((w_rot - wr) / 2, (h_rot - hr) / 2, (w_rot - wr) / 2 + wr, (h_rot - hr) / 2 + hr))
        Image.fromarray(np.uint8(rotated_img)).save(save_image_path + filename.replace('.jpg', '_{}.png'.format(angle)))


def batch_production(l, m, deg, img_num, source_folder, save_image_path, width, height, angle=0, is_show=False,
                     is_save=False):
    # 批量生产十七种对称图形
    # img_num: 产生每种图形数量

    info1 = [l, 1]  # 正方形
    info2 = [l, m, 2]  # 长方形
    info3 = [l, m, deg, 3]  # 平行四边形
    info4 = [l, deg, 4]  # 菱形
    info4_3 = [l, l, 60, 3]  # 等效菱形的平行四边形
    info5 = [l, 5]  # 正六边形

    # 三角形
    y = int(l * np.cos(deg * np.pi / 180)) * 2
    info6_1_1 = [y, l, deg, 6]  # 等腰三角形

    y = int(l * np.cos(30 * np.pi / 180)) * 2
    info6_1 = [y, l, 30, 6]  # 等腰30度三角形

    info6_1_2 = [l, l, 90, 6]  # 等腰直角三角形

    x = int(l * np.cos(deg * np.pi / 180))
    info6_2 = [x, l, deg, 6]  # 直角三角形

    x = int(l * np.cos(45 * np.pi / 180)) * 2
    info6_2_1 = [x, l, 45, 6]  # 45度等腰直角三角形

    x = int(l * np.cos(60 * np.pi / 180))
    info6_2_2 = [x, l, 60, 6]  # 直角60三角形

    info6_3 = [l, l, 60, 6]  # 等边三角形

    # 生成点
    point1 = Point(info=info1).run()
    point2 = Point(info=info2).run()
    point3 = Point(info=info3).run()
    point4 = Point(info=info4).run()
    point4_3 = Point(info=info4_3).run()
    point5 = Point(info=info5).run()
    point6_1_1 = Point(info=info6_1_1).run()
    point6_1 = Point(info=info6_1).run()
    point6_1_2 = Point(info=info6_1_2).run()
    point6_2 = Point(info=info6_2).run()
    point6_2_1 = Point(info=info6_2_1).run()
    point6_2_2 = Point(info=info6_2_2).run()
    point6_3 = Point(info=info6_3).run()

    # 图像数据
    files = os.listdir(source_folder)
    # files.sort(key=lambda x: int(x[:-4]))
    random.shuffle(files)
    for filename in files[0:img_num]:
        source_image_path = os.path.join(source_folder, filename)
        kwargs = {'source_image_path': source_image_path, 'is_show': is_show, 'is_save': is_save,
                  'save_image_path': save_image_path, 'picture_width': width, 'picture_height': height, 'angle': angle}

        P1(**kwargs, point=point1, info=info1).generate()
        P1(**kwargs, point=point2, info=info2).generate()
        P1(**kwargs, point=point3, info=info3).generate()
        P1(**kwargs, point=point4, info=info4).generate()
        P1(**kwargs, point=point5, info=info5).generate()

        P2(**kwargs, point=point2, info=info2).generate()
        P2(**kwargs, point=point3, info=info3).generate()
        P2(**kwargs, point=point6_1_1, info=info6_1_1).generate()
        P2Hexagonal(**kwargs, point=point6_3, info=info6_3).generate()

        Pm(**kwargs, point=point1, info=info1).generate()
        Pm(**kwargs, point=point2, info=info2).generate()

        Pg(**kwargs, point=point2, info=info2).generate()

        Cm(**kwargs, point=point6_1, info=info6_1).generate()

        Pmm(**kwargs, point=point1, info=info1).generate()
        Pmm(**kwargs, point=point2, info=info2).generate()

        Pmg(**kwargs, point=point2, info=info2).generate()

        Pgg(**kwargs, point=point6_1_1, info=info6_1_1).generate()

        Cmm(**kwargs, point=point6_2, info=info6_2, type='rhombic').generate()
        Cmm(**kwargs, point=point6_2_1, info=info6_2_1, type='square').generate()

        P4(**kwargs, point=point1, info=info1).generate()
        P4m(**kwargs, point=point6_1_2, info=info6_1_2).generate()
        P4g(**kwargs, point=point6_1_2, info=info6_1_2).generate()

        P3(**kwargs, point=point4_3, info=info4_3).generate()
        P3m1(**kwargs, point=point6_3, info=info6_3).generate()
        P31m(**kwargs, point=point6_1, info=info6_1).generate()

        P6(**kwargs, point=point6_3, info=info6_3).generate()
        P6m(**kwargs, point=point6_2_2, info=info6_2_2).generate()


if __name__ == '__main__':
    """
        source_folder 原文件目录  save_folder 保存文件目录
    """
    source_folder = '../images/ori_img/3/'
    save_folder = '../images/output2'

    # sum: epoch * img_num
    epoch = 1
    img_num = 1

    for i in range(epoch):
        np.random.seed(0)
        width, height = np.random.randint(900, 950, 2)  # 画布大小

        # cell：两边长一夹角（l,m,deg）
        l = 100
        m = 60
        # l = np.random.randint(60, 115)  # 边长
        # m = np.random.randint(60, 115)  # 边长
        deg = np.random.randint(40, 70)

        # 整张图片的旋转
        # angle = np.random.randint(0, 360)
        angle = 0

        # produce_None(source_folder, save_image_path + 'None/0/', img_num, angle)  # 第18类
        print('random value,  [l:{} m:{}], deg:{}, angle:{}, [w:{},h:{}]'.format(l, m, deg, angle, width, height))
        # 批量生产
        batch_production(l, m, deg, img_num, source_folder, save_folder, width=width, height=height, angle=angle,
                         is_show=False, is_save=True)
