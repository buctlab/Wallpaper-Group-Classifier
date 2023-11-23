from base.seventeen_wallpaper_line import *


def draw_line(l, m, deg, source_image_path, save_image_path, is_show):
    # 生成带标识的图
    info3 = [l, m, deg, 3]  # 平行四边形
    point3 = Point(info=info3).run()

    info4 = [l, 60, 4]  # 菱形
    point4 = Point(info=info4).run()

    info4_3 = [l, l, 60, 3]  # 模仿菱形的平行四边形
    point4_3 = Point(info=info4_3).run()

    info5 = [l, 5]  # 正六边形
    point5 = Point(info=info5).run()

    y = int(l * np.cos(30 * np.pi / 180)) * 2
    info6_1 = [y, l, 30, 6]  # 等腰30度三角形
    point6_1 = Point(info=info6_1).run()

    info6_1_2 = [l, l, 90, 6]  # 等腰直角三角形
    point6_1_2 = Point(info=info6_1_2).run()

    x = int(l * np.cos(deg * np.pi / 180))
    info6_2 = [x, l, deg, 6]  # 直角三角形
    point6_2 = Point(info=info6_2).run()

    x = int(l * np.cos(45 * np.pi / 180)) * 2
    info6_2_1 = [x, l, 45, 6]  # 45度等腰直角三角形
    point6_2_1 = Point(info=info6_2_1).run()

    x = int(l * np.cos(60 * np.pi / 180))
    info6_2_2 = [x, l, 60, 6]  # 直角60三角形
    point6_2_2 = Point(info=info6_2_2).run()

    print('random value:', l, m, deg)

    info6_3 = [l, l, 60, 6]  # 等边三角形
    point6_3 = Point(info=info6_3).run()

    info1 = [l, 1]  # 正方形
    point1 = Point(info=info1).run()
    info2 = [l, m, 2]  # 长方形
    point2 = Point(info=info2).run()

    kwargs = {'source_image_path': source_image_path, 'save_image_path': save_image_path, 'is_show': is_show,
              'picture_width': width, 'picture_height': height}

    P1Line(point=point1, info=info1, **kwargs).generate()
    P1Line(point=point2, info=info2, **kwargs).generate()
    P1Line(point=point3, info=info3, **kwargs).generate()
    P1Line(point=point4, info=info4, **kwargs).generate()
    P1Line(point=point5, info=info5, **kwargs).generate()

    P2Line(point=point2, info=info2, **kwargs).generate()
    P2Line(point=point3, info=info3, **kwargs).generate()
    P2Line(point=point6_1, info=info6_1, **kwargs).generate()
    P2HexagonalLine(point=point6_3, info=info6_3, **kwargs).generate()

    PmLine(point=point1, info=info1, **kwargs).generate()
    PmLine(point=point2, info=info2, **kwargs).generate()
    PgLine(point=point2, info=info2, **kwargs).generate()
    CmLine(point=point6_1, info=info6_1, **kwargs).generate()
    PmmLine(point=point1, info=info1, **kwargs).generate()
    PmmLine(point=point2, info=info2, **kwargs).generate()
    PmgLine(point=point2, info=info2, **kwargs).generate()
    PggLine(point=point6_1, info=info6_1, **kwargs).generate()

    CmmLine(point=point6_2, info=info6_2, **kwargs).generate()
    CmmSquareLine(point=point6_2_1, info=info6_2_1, **kwargs).generate()

    P4Line(point=point1, info=info1, **kwargs).generate()
    P4mLine(point=point6_1_2, info=info6_1_2, **kwargs).generate()
    P4gLine(point=point6_1_2, info=info6_1_2, **kwargs).generate()

    P3Line(point=point4_3, info=info4_3, **kwargs).generate()
    P3m1Line(point=point6_3, info=info6_3, **kwargs).generate()
    P31mLine(point=point6_1, info=info6_1, **kwargs).generate()

    P6Line(point=point6_3, info=info6_3, **kwargs).generate()
    P6mLine(point=point6_2_2, info=info6_2_2, **kwargs).generate()


if __name__ == '__main__':
    """
        source_image_path 原图片  save_folder 保存文件目录
    """
    source_image_path = '../images/ori_img/shao.png'
    save_folder = '../images/output_with_line'

    # 最后呈现的画布大小
    width, height = 800, 800

    # cell：两边长一夹角（l,m,deg）
    l = 100
    m = int(l * 0.618)
    deg = 60

    # is_show: 是否在运行中展示
    draw_line(l, m, deg, source_image_path, save_folder, is_show=False)
