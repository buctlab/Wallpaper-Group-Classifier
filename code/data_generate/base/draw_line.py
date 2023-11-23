import cv2
import numpy as np
from PIL import Image


# 镜像
def refelection(img, point, color):
    cv2.line(img, point[0], point[1], color, 2)


def glide_refelection(img, point, color):
    cv2.line(img, point[0], point[1], color, 1)


# 菱形点
def rhomb(img, point, color):
    # (x,y)->(x-1,y),(x+1,y),(x,y-2),(x,y+2) array[[],[]]
    points = np.array(
        [[[point[0] - 4, point[1]], [point[0], point[1] - 4], [point[0] + 4, point[1]], [point[0], point[1] + 4]]])
    # print(points)
    cv2.polylines(img, points, True, color)
    cv2.fillPoly(img, points, color)


# 正方形点
def square(img, point, color):
    cv2.rectangle(img, (point[0] - 4, point[1] - 4), (point[0] + 4, point[1] + 4), color, -1)


# 三角形点
def tri(img, point, color):
    points = np.array(
        [[[point[0], point[1] - 4], [point[0] - 4, point[1] + 4], [point[0] + 4, point[1] + 4]]])
    cv2.polylines(img, points, True, color)
    cv2.fillPoly(img, points, color)


# 六边形
def hex(img, point, color):
    points = np.array(
        [[[point[0] - 2, point[1] - 3], [point[0] - 4, point[1]], [point[0] - 2, point[1] + 2],
          [point[0] + 2, point[1] + 2], [point[0] + 4, point[1]], [point[0] + 2, point[1] - 3]]])
    cv2.polylines(img, points, True, color)
    cv2.fillPoly(img, points, color)


def draw(img, draw_list):
    # 画直线 cv2.line(img,(,),(,),(255,0,0),5)
    # 画矩阵 cv2.rectangle(img,(s,t),(s+1,t+1),(255,0,0),-1) 左上角右下角坐标
    _, _, _, alpha = img.split()
    img = np.array(img.convert('RGB'))
    for func, *parameter in draw_list:
        # print('func', func, *parameter)
        eval(func)(img, *parameter)
    img = Image.fromarray(img)
    img.putalpha(alpha)
    return img
