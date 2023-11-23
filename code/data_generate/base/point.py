import numpy as np
from base.cell import Cell


class Point:
    '''
        kind=1： 正方形 square  info[w]:一个参数-->正方形边长
        kind=2： 长方形 rectangular   info[w,h]:两个参数--> 长方形 长 宽
        kind=3： 平行四边形  oblique   info[w_1,w_2,theta]:三个参数--> 长边 短边 夹角
        kind=4： 菱形   rhombic   info[w,theta]:两个参数--> 边长 夹角
        kind=5： 正六边形  hexagon   info[w]:一个参数--> 边长
        kind=6: 三角形   triangle    info[w_1,w_2,theta]:三个参数--> 两条边及其夹角
    '''

    def __init__(self, info):
        self.kind = info[-1]
        self.info = info[0:-1]

    @staticmethod
    def square(info):
        # 正方形
        length = info[0]

        # 重新定义画布大小为正方形边长
        # rectangle_width = length
        # rectangle_height = length
        point_for_cut = np.array([[0, 0], [0, length], [length, length], [length, 0]])

        return point_for_cut

    #
    @staticmethod
    def rectangular(info):
        # 长方形
        w = info[0]
        h = info[1]

        point_for_cut = np.array(
            [[0, 0], [0, h], [w, h], [w, 0]])

        return point_for_cut

    @staticmethod
    def oblique(info):
        # 平行四边形

        # long_side 长边
        # short_side  短边
        # long_side, short_side, theta = info[0], info[1], info[2]
        long_side, short_side, theta = info[:]

        if theta >= 90 or theta <= 0:
            raise ValueError("the theta should between 90 and 0")

        rectangle_height = int(long_side * np.sin(theta * np.pi / 180))
        temp_w = int(long_side * np.cos(theta * np.pi / 180))

        rectangle_width = temp_w + short_side

        # print('point.py',short_side, long_side, rectangle_width, rectangle_height)
        point_for_cut = np.array(
            [[temp_w, 0], [0, rectangle_height], [short_side, rectangle_height],
             [temp_w + short_side, 0]])

        return point_for_cut

    @staticmethod
    def rhombic(info):
        # 菱形

        # length, theta = info[0],info[1]
        length, theta = info[:]

        if theta >= 180 or theta <= 0:
            raise ValueError("the theta should between 180 and 0")

        temp_theta = theta / 2

        temp_h = int(length * np.sin(temp_theta * np.pi / 180))
        temp_w = int(length * np.cos(temp_theta * np.pi / 180))

        rectangle_height = temp_h * 2
        rectangle_width = temp_w * 2
        point_for_cut = np.array([[temp_w, 0], [0, temp_h], [temp_w, rectangle_height], [rectangle_width, temp_h]])

        return point_for_cut

    @staticmethod
    def hexagon(info):
        # 正六边形

        length = info[0]

        temp_h = int(length * np.sin(np.pi / 3))
        # temp_w = int(length * np.cos(np.pi / 3))
        temp_w = int(length // 2)

        rectangle_height = temp_h * 2
        rectangle_width = temp_w * 2 + length

        # print(rectangle_width, rectangle_height)
        point_for_cut = np.array([[temp_w, 0], [0, temp_h], [temp_w, rectangle_height],
                                  [temp_w + length, temp_h * 2], [temp_w * 2 + length, temp_h],
                                  [temp_w + length, 0]])
        # print('point:', point_for_cut)
        return point_for_cut

    @staticmethod
    def triangle(info):
        # 三角形
        long_side, short_side, theta = info[:]
        if theta == 90:
            # 直角
            rectangle_width = long_side
            rectangle_height = short_side
            point_for_cut = np.array([[0, 0], [0, rectangle_height], [rectangle_width, rectangle_height]])

        elif theta > 0 and theta < 90:
            # 锐角
            rectangle_width = long_side
            rectangle_height = int(short_side * np.sin(theta * np.pi / 180))
            temp_w = int(short_side * np.cos(theta * np.pi / 180))
            # print(temp_w, rectangle_height, rectangle_width)
            point_for_cut = np.array([[temp_w, 0], [0, rectangle_height], [rectangle_width, rectangle_height]])

        elif theta > 90 and theta < 180:
            # 钝角
            rectangle_height = int(short_side * np.sin((theta) * np.pi / 180))
            temp_w = int(short_side * np.cos((180 - theta) * np.pi / 180))
            rectangle_width = long_side + temp_w

            # print(temp_w, rectangle_height, rectangle_width)
            point_for_cut = np.array([[0, 0], [temp_w, rectangle_height], [rectangle_width, rectangle_height]])

        else:
            raise ValueError("the theta should between 180 and 0")

        return point_for_cut

    def run(self):
        if self.kind == 1:
            # return None
            return self.square(self.info)
        elif self.kind == 2:
            # return None
            return self.rectangular(self.info)
        elif self.kind == 3:
            return self.oblique(self.info)
        elif self.kind == 4:
            return self.rhombic(self.info)
        elif self.kind == 5:
            return self.hexagon(self.info)
        elif self.kind == 6:
            return self.triangle(self.info)
        else:
            raise ValueError("kind should between 1 and 6")


# test
if __name__ == '__main__':
    print('start', np.cos(np.pi / 3))

    source_image_path = '../images/dataset/16.png'

    # info = [300, 200, 70, 3]
    # info = [50,50,120,6]

    x = 100
    theta = 30
    y = int(x * np.cos(theta * np.pi / 180)) * 2
    info = [x, y, theta, 6]
    print(info[-1], info[0:-1])
    # point, rectangle_width, rectangle_height  = Point().square(info=[100])  # 正方形
    # point, rectangle_width, rectangle_height = Point().rectangular(info=[100, 200])  # 长方形
    point, rectangle_width, rectangle_height = Point(info=info).run()  # 平行四边形
    # point, rectangle_width, rectangle_height = Point().rhombic(info=[200, 60])  # 菱形
    # point, rectangle_width, rectangle_height = Point().hexagon(info=[50])  # 正六边形
    # point, rectangle_width, rectangle_height = Point().triangle(info=[200, 100, 150])  # 三角形

    cell = Cell(rectangle_width=rectangle_width, rectangle_height=rectangle_height, point=point,
                source_image_path=source_image_path, kind=3, picture_width=600, picture_height=600)
    cell.paste_img((0, 0))
    cell.show()

    cell.over_spread(point[0])
