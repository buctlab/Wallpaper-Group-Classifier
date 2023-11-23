from base.cell import Cell
from numpy import tan, pi, array
from PIL import Image
import os


def rotate_operate(cell, deg, r, num):
    '''
    path 图片路径
    直角三角形
    num 旋转次数
    角度 deg = 360 / num
    边长 length 即 圆半径 r
    粘贴在大画布上
    '''
    cell.paste_img((r, 0))
    cell.update_img()

    for i in range(1, num):
        cell.rotate(-i * deg, (0, 0), center=(r, r), expand=False)

    return cell.markImg


def mirror_operate(cell, deg, num):
    # 　镜像
    cell.paste_img((r, 0))
    cell.update_img()
    cell.lr_mirror()
    cell.img.show()
    cell.paste_img((0, 0))
    cell.update_img()
    cell.img.show()
    for i in range(num // 2):
        cell.rotate(deg * 2, expand=False)
        cell.paste_img((0, 0))
    cell.markImg.show()


# 切割圆形图像
def circle(img, img_name, save_path, color, num):
    cir_path = save_path + '/' + str(num)
    cir_fig_name = color + '_' + str(num) + '_cir_' + img_name + '.png'
    if not os.path.exists(cir_path):
        os.makedirs(cir_path)
    size = img.size
    # print(size)

    # 因为是要圆形，所以需要正方形的图片
    r2 = min(size[0], size[1])
    if size[0] != size[1]:
        ima = img.resize((r2, r2), Image.ANTIALIAS)

    # r3 圆的半径
    r3 = int(r2 / 2)
    imb = Image.new('RGBA', (r3 * 2, r3 * 2), (255, 255, 255, 0))
    pima = img.load()  # 像素的访问对象
    pimb = imb.load()
    r = float(r2 / 2)  # 圆心横坐标

    for i in range(r2):
        for j in range(r2):
            lx = abs(i - r)  # 到圆心距离的横坐标
            ly = abs(j - r)  # 到圆心距离的纵坐标
            l = (pow(lx, 2) + pow(ly, 2)) ** 0.5  # 三角函数 半径
            if l < r3:
                pimb[i - (r - r3), j - (r - r3)] = pima[i, j]

    markImg = Image.new('RGBA', imb.size, color)
    _, _, _, mask = imb.split()
    markImg.paste(imb, (0, 0), mask=mask)
    markImg.save(os.path.join(cir_path, cir_fig_name))
    return cir_path


def one_run(folder, name, save_path, r, num, color, point, deg):
    img_path = folder + '/' + name
    canvas_w, canvas_h = 2 * r, 2 * r  # 画布大小

    cell = Cell(source_image_path=img_path, rectangle_width=canvas_w, rectangle_height=canvas_h, point=point)
    img = rotate_operate(cell, deg, r, num)
    circle(img, name, save_path, color, num)


def batch_produce(folder, save_path, r, num, color, batch=10):
    # 批量成产
    path_list = os.listdir(folder)
    # 生成坐标点
    deg = 360 / num
    if num == 2:
        l = r
        point = array([[0, 0], [2 * r, 0], [2 * r, 2 * r], [0, 2 * r]])
    elif num == 4:
        l = 0
        point = array([[0, 0], [r, 0], [r, r], [0, r]])
    elif num < 4:
        # 生成四边形
        l = int(r * tan(((deg - 90) / 180) * pi))
        point = array([[0, 0], [r, 0], [r, r + l], [0, r]])
    else:
        # 生成三角形点
        l = int(r * tan((deg / 180) * pi))
        point = array([[0, 0], [0, r], [l, 0]])
    print('l : {}, r : {}, deg : {} color : {}'.format(l, r, deg, color))

    for name in path_list[:batch]:
        one_run(folder, name, save_path, r, num, color, point, deg)


'''
num = 12 deg = 30
num = 8 deg = 45
num = 6 deg = 60
num = 5 deg = 72
'''

if __name__ == '__main__':
    print('start')
    # r = int(input('please input r:'))  # 半径
    # num = int(input('please input the number of lines cut:'))  # 旋转个数
    # color = input('please input bg color:(black/white)')  # 背景颜色

    r = 100  # 半径
    num = [2, 3, 4, 6]  # 旋转个数
    color = ['black', 'white']  # 背景颜色
    batch_size = 20  # 一批数据的数量

    # run('E:/jupyter_tree/data/VOCdevkit/VOC2012/JPEGImages/2012_000003.jpg', r, num, color)

    img_folder = '../images/ori_img'
    save_path = '../images/cir_images'
    for n in num:
        for c in color:
            batch_produce(img_folder, save_path, r, n, c, batch=batch_size)
