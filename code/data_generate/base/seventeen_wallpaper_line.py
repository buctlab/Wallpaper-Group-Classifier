from base.cell import Cell
from base.point import Point
from base.draw_line import *
from base.seventeen_wallpaper import *

# colors
black = (0, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
red = (255, 0, 0)
grey = (190, 190, 190)
yellow = (255, 255, 0)
purple = (160, 32, 240)
pink = (255, 192, 203)
orange = (255, 165, 0)


class P1Line(P1):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(self):
        func_list = [
            ('border',),
            ('paste_img', (0, 0)),
            ('over_spread', self.point[0]),
            ('save',),
            ('show',)
        ]
        self.run(func_list=func_list)

    def save(self):
        if self.is_save:
            path_name_min = self.save_image_path + '/p1_' + str(self.info[-1]) + '_min.png'
            path_name = self.save_image_path + '/p1_' + str(self.info[-1]) + '_result.png'

            self.markImg.save(path_name_min)
            self.GroundImg.save(path_name)


class P2Line(P2):
    def __init__(self, **kwargs):
        super(P2Line, self).__init__(**kwargs)

    def generate(self):
        func_list = [('paste_img', (0, 0)),
                     ('update_img',),
                     ('rotate', -180, (0, 0))
                     ]
        if self.info[-1] == 3:
            func_list[0] = ('paste_img', (0, self.rectangle_height // 2))

        self.run(func_list=func_list)

        w, h = self.rectangle_width, self.rectangle_height
        draw_list = [
            # 边框
            ('glide_refelection', ((0, 0), (0, self.point[1][1] * 2)), grey),
            ('glide_refelection', ((0, 0), (self.point[2][0] * 2, 0)), grey),

            # 旋转中心点
            ('rhomb', (0, 0), red),
            ('rhomb', (w, 0), red),
            ('rhomb', (0, h), red),
            ('rhomb', (w, h), red),

            ('rhomb', (w // 2, h // 2), blue),

            ('rhomb', (w // 2, 0), pink),
            ('rhomb', (w // 2, h), pink),

            ('rhomb', (0, h // 2), green),
            ('rhomb', (w, h // 2), green),
        ]

        if self.info[-1] == 6:
            draw_list = [
                # 边框
                ('glide_refelection', ((w // 2, 0), (0, h // 2)), grey),
                ('glide_refelection', ((w // 2 - 1, 0), (w, h // 2)), grey),

                # 旋转中心点
                ('rhomb', (w // 2, 0), red),
                ('rhomb', (w // 2, h), red),
                ('rhomb', (0, h // 2), red),
                ('rhomb', (w, h // 2), red),

                ('rhomb', (w // 2, h // 2), blue),

                ('rhomb', (int(w * (3 / 4)), h // 4), pink),
                ('rhomb', (w // 4, int(h * (3 / 4))), pink),

                ('rhomb', (int(w * (3 / 4)), int(h * (3 / 4))), green),
                ('rhomb', (w // 4, h // 4), green),
            ]
        elif self.info[-1] == 3:
            draw_list = [
                # 边框
                ('glide_refelection', ((self.point[0][0] * 2, 0), (0, h)), grey),
                ('glide_refelection', ((self.point[0][0] * 2, 0), (w, 0)), grey),

                # 旋转中心点
                ('rhomb', (w // 2, h // 2), blue),

                ('rhomb', (self.point[0][0], h // 2), green),
                ('rhomb', (w - self.point[0][0], h // 2), green),

                ('rhomb', (self.point[0][0] * 2 + self.info[1] // 2, 0), pink),
                ('rhomb', (self.info[1] // 2, h), pink),

                ('rhomb', (self.point[0][0] * 2, 0), red),
                ('rhomb', (0, h), red),
                ('rhomb', (w, 0), red),
                ('rhomb', (self.info[1], h), red)
            ]

        self.markImg = draw(self.markImg, draw_list)
        func_list = [
            ('over_spread',),
            ('save',),
            ('show',)
        ]
        if self.kind == 3:
            func_list[0] = ('over_spread', (self.point[0][0] * 2, 0))
        self.run(func_list=func_list)

    def save(self):
        if self.is_save:
            path_name_min = self.save_image_path + '/p2_' + str(self.info[-1]) + '_min.png'
            path_name = self.save_image_path + '/p2_' + str(self.info[-1]) + '_result.png'

            self.markImg.save(path_name_min)
            self.GroundImg.save(path_name)


class P2HexagonalLine(P2Hexagonal):
    def __init__(self, **kwargs):
        super(P2HexagonalLine, self).__init__(**kwargs)

    def generate(self):
        func_list = [('paste_img', (0, 0)),
                     ('update_img',),
                     ('rotate', -180, (0, 0))
                     ]
        self.run(func_list=func_list)
        draw_list = [
            # 边框
            ('glide_refelection', ((self.point[0][0], 0), (0, self.point[1][1])), grey),
            ('glide_refelection', ((self.point[0][0], 0), (self.point[0][0] + self.point[2][0], 0)), grey),
            ('glide_refelection', ((self.point[0][0], 0), (self.point[2][0], self.point[1][1])), grey),

            # 旋转中心点
            ('rhomb', (self.point[0][0], 0), purple),
            ('rhomb', (0, self.point[1][1]), purple),
            ('rhomb', (self.point[2][0] + self.point[0][0], 0), purple),
            ('rhomb', (self.point[2][0], self.point[1][1]), purple),

            ('rhomb', ((self.point[2][0] + self.point[0][0]) // 2, self.point[1][1] // 2), yellow),

            ('rhomb', (self.point[0][0] // 2, self.point[1][1] // 2), green),
            ('rhomb', (self.point[2][0] + self.point[0][0] // 2, self.point[1][1] // 2), green),

            ('rhomb', (self.point[0][0] + self.point[2][0] // 2, 0), pink),
            ('rhomb', (self.point[2][0] // 2, self.point[1][1]), pink),

        ]
        self.markImg = draw(self.markImg, draw_list)
        func_list = [
            ('over_spread', (self.point[0][0], 0)),
            ('save',),
            ('show',)
        ]
        self.run(func_list=func_list)

    def save(self):
        if self.is_save:
            self.markImg.save(self.save_image_path + '/p2_hexagonal_min.png')
            self.GroundImg.save(self.save_image_path + '/p2_hexagonal_result.png')


class PmLine(Pm):
    def __init__(self, **kwargs):
        super(PmLine, self).__init__(**kwargs)

    def generate(self):
        func_list = [('paste_img', (0, 0)),
                     ('update_img',),
                     ('lr_mirror', (0, 0))
                     ]
        if self.info[-1] == 2:
            func_list[2] = ('tb_mirror', (0, 0))
        else:
            func_list = func_list

        self.run(func_list=func_list)
        draw_list = [
            # 边框
            ('glide_refelection', ((0, 0), (0, self.point[1][1] * 2)), grey),
            ('glide_refelection', ((0, 0), (self.point[2][0] * 2, 0)), grey),

            #
            ('refelection', ((0, 0), (0, self.point[1][1] * 2)), blue),
            ('refelection', ((self.point[2][0] * 2, 0), (self.point[2][0] * 2, self.point[1][1] * 2)), blue),

            ('refelection', ((self.point[2][0], 0), (self.point[2][0], self.point[1][1] * 2)), red)
        ]

        if self.info[-1] == 2:
            draw_list[2] = ('refelection', ((0, 0), (self.point[2][0], 0)), blue)
            draw_list[3] = ('refelection', ((0, self.point[1][1] * 2), (self.point[2][0], self.point[1][1] * 2)), blue)
            draw_list[4] = ('refelection', ((0, self.point[1][1]), (self.point[2][0], self.point[1][1])), red)
        else:
            draw_list = draw_list

        self.markImg = draw(self.markImg, draw_list)
        func_list = [
            ('over_spread',),
            ('save',),
            ('show',)
        ]
        self.run(func_list=func_list)

    def save(self):
        if self.is_save:
            if self.info[-1] == 2:
                path_name_min = self.save_image_path + '/pm_' + str(self.info[-1]) + '_min.png'
                path_name = self.save_image_path + '/pm_' + str(self.info[-1]) + '_result.png'
            else:
                path_name_min = self.save_image_path + '/pm_min.png'
                path_name = self.save_image_path + '/pm_result.png'

            self.markImg.save(path_name_min)
            self.GroundImg.save(path_name)


class PgLine(Pg):
    def __init__(self, **kwargs):
        super(PgLine, self).__init__(**kwargs)

    def generate(self):
        func_list = [('paste_img', (0, 0)),
                     ('update_img',),
                     ('lr_mirror', (0, self.point[1][1]))
                     ]
        self.run(func_list=func_list)
        draw_list = [
            # 边框
            ('glide_refelection', ((0, 0), (self.point[2][0], 0)), grey),
            ('glide_refelection', ((0, 0), (0, self.point[1][1] * 2)), grey),

            #  glide refelection
            ('glide_refelection', ((0, 0), (0, self.point[1][1] * 2)), red),
            ('glide_refelection', ((self.point[2][0], 0), (self.point[2][0], self.point[1][1] * 2)), red),

            ('glide_refelection', ((self.point[2][0] // 2, 0), (self.point[2][0] // 2, self.point[1][1] * 2)), green),

        ]
        self.markImg = draw(self.markImg, draw_list)
        func_list = [
            ('over_spread',),
            ('save',),
            ('show',)
        ]
        self.run(func_list=func_list)

    def save(self):
        if self.is_save:
            self.markImg.save(self.save_image_path + '/pg_min.png')
            self.GroundImg.save(self.save_image_path + '/pg_result.png')


class CmLine(Cm):
    def __init__(self, **kwargs):
        super(CmLine, self).__init__(**kwargs)

    def generate(self):
        func_list = [('paste_img', (0, 0)),
                     ('update_img',),
                     ('tb_mirror', (0, 0))
                     ]
        self.run(func_list=func_list)
        draw_list = [
            # 边框
            ('glide_refelection', ((self.point[2][0] // 2, 0), (0, self.point[1][1])), grey),
            ('glide_refelection', ((self.point[2][0] // 2, 0), (self.point[2][0], self.point[1][1])), grey),

            # 镜像
            ('refelection', ((0, self.point[1][1]), (self.point[2][0], self.point[1][1])), blue),

            #  glide refelection
            ('glide_refelection', ((self.point[2][0] // 4, self.point[1][1] // 2),
                                   (int(self.point[2][0] * (3 / 2)), self.point[1][1] // 2)), green),
            ('glide_refelection', ((self.point[2][0] // 4, int(self.point[1][1] * (3 / 2))),
                                   (int(self.point[2][0] * (3 / 2)), int(self.point[1][1] * (3 / 2)))), green),

        ]
        self.markImg = draw(self.markImg, draw_list)
        func_list = [
            ('over_spread',),
            ('save',),
            ('show',)
        ]
        self.run(func_list=func_list)

    def save(self):
        if self.is_save:
            self.markImg.save(self.save_image_path + '/cm_min.png')
            self.GroundImg.save(self.save_image_path + '/cm_result.png')


class PmmLine(Pmm):
    def __init__(self, **kwargs):
        super(PmmLine, self).__init__(**kwargs)

    def generate(self):
        print('kind', self.kind)
        func_list = [('paste_img', (0, 0)),
                     ('update_img',),
                     ('lr_mirror', (0, 0)),
                     ('update_img',),
                     ('tb_mirror', (0, 0))
                     ]
        self.run(func_list=func_list)
        draw_list = [
            # 边框
            ('glide_refelection', ((0, 0), (self.point[2][0] * 2, 0)), grey),
            ('glide_refelection', ((0, 0), (0, self.point[1][1] * 2)), grey),

            # 镜像
            ('refelection', ((self.point[2][0], 0), (self.point[2][0], self.point[1][1] * 2)), green),
            ('refelection', ((0, self.point[1][1]), (self.point[2][0] * 2, self.point[1][1])), orange),
            ('refelection', ((0, 0), (0, self.point[1][1] * 2)), pink),
            ('refelection', ((self.point[2][0] * 2, 0), (self.point[2][0] * 2, self.point[1][1] * 2)), pink),
            ('refelection', ((0, 0), (self.point[2][0] * 2, 0)), blue),
            ('refelection', ((0, self.point[1][1] * 2), (self.point[2][0] * 2, self.point[1][1] * 2)), blue),

            #  旋转中心点
            ('rhomb', (self.point[2][0], self.point[1][1]), yellow),

            ('rhomb', (0, 0), purple),
            ('rhomb', (0, self.point[1][1] * 2), purple),
            ('rhomb', (self.point[2][0] * 2, 0), purple),
            ('rhomb', (self.point[2][0] * 2, self.point[1][1] * 2), purple),

            ('rhomb', (0, self.point[1][1]), green),
            ('rhomb', (self.point[2][0] * 2, self.point[1][1]), green),

            ('rhomb', (self.point[2][0], 0), pink),
            ('rhomb', (self.point[2][0], self.point[1][1] * 2), pink),

        ]
        self.markImg = draw(self.markImg, draw_list)
        func_list = [
            ('over_spread',),
            ('save',),
            ('show',)
        ]
        self.run(func_list=func_list)

    def save(self):
        if self.is_save:
            self.markImg.save(self.save_image_path + '/pmm_min.png')
            self.GroundImg.save(self.save_image_path + '/pmm_result.png')


class PmgLine(Pmg):
    def __init__(self, **kwargs):
        super(PmgLine, self).__init__(**kwargs)

    def generate(self):
        func_list = [('paste_img', (0, 0)),
                     ('rotate', 180, (self.info[0], 0)),
                     ('update_img',),
                     ('tb_mirror', (0, 0))
                     ]
        self.run(func_list=func_list)
        draw_list = [
            # 边框
            ('glide_refelection', ((0, 0), (self.point[3][0] * 2, 0)), grey),
            ('glide_refelection', ((0, 0), (0, self.point[1][1] * 2)), grey),

            # 镜像
            ('refelection', ((0, 0), (self.point[3][0] * 2, 0)), blue),
            ('refelection', ((0, self.point[1][1]), (self.point[3][0] * 2, self.point[1][1])), blue),
            ('refelection', ((0, self.point[1][1] * 2), (self.point[3][0] * 2, self.point[1][1] * 2)), blue),

            ('glide_refelection', ((0, 0), (0, self.point[1][1] * 2)), red),
            ('glide_refelection', ((self.point[3][0], 0), (self.point[3][0], self.point[1][1] * 2)), blue),

            #  旋转中心点
            ('rhomb', (self.point[3][0], self.point[1][1] // 2), yellow),
            ('rhomb', (self.point[3][0], int(self.point[1][1] * (3 / 2))), yellow),

            ('rhomb', (0, self.point[1][1] // 2), purple),
            ('rhomb', (0, int(self.point[1][1] * (3 / 2))), purple),
            ('rhomb', (self.point[3][0] * 2, self.point[1][1] // 2), purple),
            ('rhomb', (self.point[3][0] * 2, int(self.point[1][1] * (3 / 2))), purple),

        ]
        self.markImg = draw(self.markImg, draw_list)
        func_list = [
            ('over_spread',),
            ('save',),
            ('show',)
        ]
        self.run(func_list=func_list)

    def save(self):
        if self.is_save:
            self.markImg.save(self.save_image_path + '/pmg_min.png')
            self.GroundImg.save(self.save_image_path + '/pmg_result.png')


class PggLine(Pgg):
    def __init__(self, **kwargs):
        super(PggLine, self).__init__(**kwargs)

    def generate(self):
        func_list = [('paste_img', (0, 0)),
                     ('tb_mirror', (self.point[0][0], 0)),
                     ('tb_mirror', (-self.point[0][0], 0)),
                     ('update_img',),
                     ('rotate', 180, (0, 0))
                     ]
        self.run(func_list=func_list)
        draw_list = [
            # 边框
            ('glide_refelection', ((self.point[2][0] // 2, 0), (0, self.point[1][1])), grey),
            ('glide_refelection', ((self.point[2][0] // 2, 0), (self.point[2][0], self.point[1][1])), grey),
            ('glide_refelection', ((0, self.point[1][1]), (self.point[2][0] // 2, self.point[1][1] * 2)), grey),
            ('glide_refelection', ((self.point[2][0] // 2, self.point[1][1] * 2),
                                   (self.point[2][0], self.point[1][1])), grey),
            ('glide_refelection', ((0, self.point[1][1]),
                                   (self.point[2][0], self.point[1][1])), grey),

            # glide_refelection
            ('glide_refelection', ((0, self.point[1][1] // 2), (self.point[2][0], self.point[1][1] // 2)), red),
            ('glide_refelection', ((0, int(self.point[1][1] * (3 / 2))),
                                   (self.point[2][0], int(self.point[1][1] * (3 / 2)))), red),

            ('glide_refelection', ((self.point[2][0] // 4, 0), (self.point[2][0] // 4, self.point[1][1] * 2)), blue),
            ('glide_refelection', ((int(self.point[2][0] * (3 / 4)), 0),
                                   (int(self.point[2][0] * (3 / 4)), self.point[1][1] * 2)), blue),

            #  旋转中心点
            ('rhomb', (self.point[2][0] // 2, 0), yellow),
            ('rhomb', (0, self.point[1][1]), yellow),
            ('rhomb', (self.point[2][0], self.point[1][1]), yellow),
            ('rhomb', (self.point[2][0] // 2, self.point[1][1] * 2), yellow),

            ('rhomb', (0, 0), purple),
            ('rhomb', (self.point[2][0], 0), purple),
            ('rhomb', (0, self.point[1][1] * 2), purple),
            ('rhomb', (self.point[2][0], self.point[1][1] * 2), purple),

            ('rhomb', (self.point[2][0] // 2, self.point[1][1]), purple),

        ]
        self.markImg = draw(self.markImg, draw_list)
        func_list = [
            ('over_spread',),
            ('save',),
            ('show',)
        ]
        self.run(func_list=func_list)

    def save(self):
        if self.is_save:
            self.markImg.save(self.save_image_path + '/pgg_min.png')
            self.GroundImg.save(self.save_image_path + '/pgg_result.png')


class CmmLine(Cmm):
    def __init__(self, **kwargs):
        kwargs.update({'type': 'rhombic'})
        super(CmmLine, self).__init__(**kwargs)

    def generate(self):
        func_list = [('paste_img', (0, 0)),
                     ('update_img',),
                     ('tb_mirror', (0, 0)),
                     ('update_img',),
                     ('lr_mirror', (0, 0))
                     ]
        self.run(func_list=func_list)

        draw_list = [
            ('glide_refelection', ((self.point[2][0], 0), (0, self.point[2][1])), grey),
            ('glide_refelection', ((self.point[2][0], 0), (self.point[2][0] * 2, self.point[2][1])), grey),
            ('glide_refelection', ((0, self.point[2][1]), (self.point[2][0], self.point[2][1] * 2)), grey),
            ('glide_refelection', ((self.point[2][0] * 2, self.point[2][1]), (self.point[2][0], self.point[2][1] * 2)),
             grey),

            ('refelection', ((0, self.point[2][1]), (self.point[2][0] * 2, self.point[2][1])), blue),
            ('refelection', ((self.point[2][0], 0), (self.point[2][0], self.point[2][1] * 2)), red),

            ('glide_refelection', ((self.point[2][0] // 2, 0), (self.point[2][0] // 2, self.point[2][1] * 2)), red),
            ('glide_refelection',
             ((int(self.point[2][0] * (3 / 2)), 0), ((int(self.point[2][0] * (3 / 2)), self.point[2][1] * 2))), red),

            ('glide_refelection', (
                (self.point[2][0] // 2, self.point[2][1] // 2),
                ((int(self.point[2][0] * (3 / 2))), self.point[2][1] // 2)),
             blue),
            ('glide_refelection', ((self.point[2][0] // 2, int(self.point[2][1] * (3 / 2))),
                                   (int(self.point[2][0] * (3 / 2)), int(self.point[2][1] * (3 / 2)))), blue),

            ('rhomb', (self.point[2][0], 0), purple),
            ('rhomb', (0, self.point[2][1]), purple),
            ('rhomb', (self.point[2][0], self.point[2][1] * 2), purple),
            ('rhomb', (self.point[2][0] * 2, self.point[2][1]), purple),

            ('rhomb', (self.point[2][0], self.point[2][1]), yellow),

            ('rhomb', (self.point[2][0] // 2, self.point[2][1] // 2), green),
            ('rhomb', (self.point[2][0] // 2, int(self.point[2][1] * (3 / 2))), green),
            ('rhomb', ((int(self.point[2][0] * (3 / 2))), self.point[2][1] // 2), green),
            ('rhomb', ((int(self.point[2][0] * (3 / 2))), int(self.point[2][1] * (3 / 2))), green),

        ]
        self.markImg = draw(self.markImg, draw_list)
        func_list = [('over_spread',),
                     ('save',),
                     ('show',)
                     ]
        self.run(func_list=func_list)

    def save(self):
        if self.is_save:
            self.markImg.save(self.save_image_path + '/cmm_min.png')
            self.GroundImg.save(self.save_image_path + '/cmm_result.png')


class CmmSquareLine(Cmm):
    def __init__(self, **kwargs):
        kwargs.update({'type': 'square'})
        super(CmmSquareLine, self).__init__(**kwargs)

    def generate(self):
        func_list = [('paste_img', (0, int(self.point[2][0] // 2))),
                     ('update_img',),
                     ('mirror_angle', -90, (0, 0)),
                     ('mirror_angle', 90, (0, 0)),
                     ('rotate', -180, (0, 0))
                     ]
        self.run(func_list=func_list)
        draw_list = [
            ('glide_refelection', ((0, 0), (0, self.point[2][0])), grey),
            ('glide_refelection', ((0, 0), (self.point[2][0], 0)), grey),

            ('refelection', ((0, self.point[2][0]), (self.point[2][0], 0)), blue),
            ('refelection', ((0, 0), (self.point[2][0], self.point[2][0])), red),

            ('glide_refelection', ((0, self.point[2][0] // 2), (self.point[2][0] // 2, self.point[2][0])), red),
            ('glide_refelection', ((self.point[2][0] // 2, 0), ((self.point[2][0], self.point[2][0] // 2))), red),

            ('glide_refelection', ((0, self.point[2][0] // 2), ((self.point[2][0] // 2, 0))), blue),
            ('glide_refelection',
             ((self.point[2][0] // 2, self.point[2][0]), ((self.point[2][0], self.point[2][0] // 2))), blue),

            #
            ('rhomb', (0, 0), purple),
            ('rhomb', (0, self.point[2][0]), purple),
            ('rhomb', (self.point[2][0], 0), purple),
            ('rhomb', (self.point[2][0], self.point[2][0]), purple),
            #
            ('rhomb', (self.point[2][0] // 2, self.point[2][0] // 2), yellow),
            #
            ('rhomb', (0, self.point[2][0] // 2), green),
            ('rhomb', (self.point[2][0] // 2, 0), green),
            ('rhomb', (self.point[2][0] // 2, self.point[2][0]), green),
            ('rhomb', (self.point[2][0], self.point[2][0] // 2), green),

        ]
        self.markImg = draw(self.markImg, draw_list)
        func_list = [('over_spread',),
                     ('save',),
                     ('show',)
                     ]
        self.run(func_list=func_list)

    def save(self):
        if self.is_save:
            self.markImg.save(self.save_image_path + '/cmm_square_min.png')
            self.GroundImg.save(self.save_image_path + '/cmm_squre_result.png')


class P4Line(P4):
    def __init__(self, **kwargs):
        super(P4Line, self).__init__(**kwargs)

    def generate(self):
        func_list = [
            ('border',),  # 小画布外边框
            ('paste_img', (0, 0)),
            ('update_img',),
            ('rotate', -90, (0, 0)),
            ('rotate', 180, (0, 0)),
            ('rotate', 90, (0, 0))
        ]

        self.run(func_list=func_list)
        draw_list = [
            # 边框
            ('glide_refelection', ((self.point[2][0], 0), (self.point[2][0], self.point[2][0] * 2)), grey),
            ('glide_refelection', ((0, self.point[2][0]), (self.point[2][0] * 2, self.point[2][0])), grey),

            #  旋转中心点
            ('rhomb', (self.point[2][0], 0), pink),
            ('rhomb', (0, self.point[2][0]), pink),
            ('rhomb', (self.point[2][0], self.point[2][0] * 2), pink),
            ('rhomb', (self.point[2][0] * 2, self.point[2][0]), pink),

            ('square', (0, 0), purple),
            ('square', (self.point[2][0] * 2, 0), purple),
            ('square', (0, self.point[2][0] * 2), purple),
            ('square', (self.point[2][0] * 2, self.point[2][0] * 2), purple),

            ('square', (self.point[2][0], self.point[2][0]), green),

        ]
        self.markImg = draw(self.markImg, draw_list)
        func_list = [
            ('over_spread',),
            ('save',),
            ('show',)
        ]
        self.run(func_list=func_list)

    def save(self):
        if self.is_save:
            self.markImg.save(self.save_image_path + '/p4_min.png')
            self.GroundImg.save(self.save_image_path + '/p4_result.png')


class P4mLine(P4m):
    def __init__(self, **kwargs):
        super(P4mLine, self).__init__(**kwargs)

    def generate(self):
        func_list = [
            ('border',),
            ('paste_img', (0, 0)),
            ('mirror_angle', 90, (0, 0)),
            ('update_img',),
            ('tb_mirror', (0, 0)),
            ('update_img',),
            ('lr_mirror', (0, 0))
        ]

        self.run(func_list=func_list)
        draw_list = [
            # 镜像
            ('refelection', ((0, 0), (self.point[2][0] * 2, self.point[1][1] * 2)), blue),
            ('refelection', ((0, self.point[1][1] * 2), (self.point[2][0] * 2, 0)), blue),
            ('refelection', ((self.point[2][0], 0), (self.point[2][0], self.point[1][1] * 2)), pink),
            ('refelection', ((0, self.point[1][1]), (self.point[2][0] * 2, self.point[1][1])), pink),

            ('refelection', ((0, 0), (self.point[2][0] * 2, 0)), orange),
            ('refelection', ((0, 0), (0, self.point[1][1] * 2)), orange),

            # glide_refelection
            ('glide_refelection', ((0, self.point[1][1]), (self.point[2][0], 0)), green),
            ('glide_refelection', ((0, self.point[1][1]), (self.point[2][0], self.point[1][1] * 2)), green),
            ('glide_refelection', ((self.point[2][0] * 2, self.point[1][1]), (self.point[2][0], self.point[1][1] * 2)),
             green),
            ('glide_refelection', ((self.point[2][0], 0), (self.point[2][0] * 2, self.point[1][1])), green),

            #  旋转中心点
            ('rhomb', (self.point[2][0], 0), red),
            ('rhomb', (0, self.point[1][1]), red),
            ('rhomb', (self.point[2][0], self.point[1][1] * 2), red),
            ('rhomb', (self.point[1][1] * 2, self.point[1][1]), red),

            ('square', (0, 0), purple),
            ('square', (self.point[2][0] * 2, 0), purple),
            ('square', (0, self.point[1][1] * 2), purple),
            ('square', (self.point[2][0] * 2, self.point[1][1] * 2), purple),

            ('square', (self.point[2][0], self.point[1][1]), yellow),

        ]
        self.markImg = draw(self.markImg, draw_list)
        func_list = [
            ('over_spread',),
            ('save',),
            ('show',)
        ]
        self.run(func_list=func_list)

    def save(self):
        if self.is_save:
            self.markImg.save(self.save_image_path + '/p4m_min.png')
            self.GroundImg.save(self.save_image_path + '/p4m_result.png')


class P4gLine(P4g):
    def __init__(self, **kwargs):
        super(P4gLine, self).__init__(**kwargs)

    def rectangle_init(self, point):
        kind = 1
        w_w = 2 * point[2][0]
        h_h = 2 * point[2][1]
        return kind, w_w, h_h

    def generate(self):
        func_list = [
            ('border',),
            ('paste_img', (0, self.point[2][1])),
            ('mirror_angle', 90, (0, self.point[2][1])),
            ('update_img',),
            ('rotate', -90, (0, 0)),
            ('rotate', -180, (0, 0)),
            ('rotate', 90, (0, 0))
        ]
        self.run(func_list=func_list)
        draw_list = [
            # 边框
            ('glide_refelection', ((0, 0), (self.point[2][0] * 2, 0)), grey),

            # 镜像
            ('refelection', ((self.point[2][0], 0), (0, self.point[1][1])), blue),
            ('refelection', ((0, self.point[1][1]), (self.point[2][0], self.point[1][1] * 2)), blue),
            ('refelection', ((self.point[2][0], 0), (self.point[2][0] * 2, self.point[1][1])), blue),
            ('refelection', ((self.point[2][0] * 2, self.point[1][1]), (self.point[2][0], self.point[1][1] * 2)), blue),

            # glide_refelection
            ('glide_refelection', ((0, 0), (self.point[2][0] * 2, self.point[1][1] * 2)), red),
            ('glide_refelection', ((self.point[2][0] * 2, 0), (0, self.point[1][1] * 2)), red),

            ('glide_refelection', ((self.point[2][0] // 2, 0), (self.point[2][0] // 2, self.point[1][1] * 2)),
             green),
            ('glide_refelection', ((int(self.point[2][0] * (3 / 2)), 0),
                                   ((int(self.point[2][0] * (3 / 2)), self.point[1][1] * 2))), green),

            ('glide_refelection', ((0, self.point[1][1] // 2), (self.point[2][0] * 2, self.point[1][1] // 2)),
             green),
            ('glide_refelection', ((0, int(self.point[1][1] * (3 / 2))),
                                   (self.point[2][0] * 2, int(self.point[1][1] * (3 / 2)))), green),

            #  旋转中心点
            ('rhomb', (self.point[2][0], 0), red),
            ('rhomb', (0, self.point[1][1]), red),
            ('rhomb', (self.point[2][0], self.point[1][1] * 2), red),
            ('rhomb', (self.point[1][1] * 2, self.point[1][1]), red),

            ('square', (0, 0), green),
            ('square', (self.point[2][0] * 2, 0), green),
            ('square', (0, self.point[1][1] * 2), green),
            ('square', (self.point[2][0] * 2, self.point[1][1] * 2), green),

            ('square', (self.point[2][0], self.point[1][1]), green),

        ]
        self.markImg = draw(self.markImg, draw_list)
        func_list = [
            ('over_spread',),
            ('save',),
            ('show',)
        ]
        self.run(func_list=func_list)

    def save(self):
        if self.is_save:
            self.markImg.save(self.save_image_path + '/p4g_min.png')
            self.GroundImg.save(self.save_image_path + '/p4g_result.png')


class P3Line(P3):
    def __init__(self, **kwargs):
        super(P3Line, self).__init__(**kwargs)

    def generate(self):
        func_list = [
            ('border',),
            ('paste_img', (self.point[0][0], self.point[1][1])),
            ('update_img',),
            ('rotate', -120, (0, 0), None, False),
            ('rotate', 120, (0, 0), None, False)
        ]

        self.run(func_list=func_list)
        draw_list = [
            # 边框
            ('glide_refelection', ((self.info[0] // 2, 0), (int(self.info[0] * (3 / 2)), 0)), grey),
            ('glide_refelection', ((self.info[0] // 2, 0), (0, self.point[1][1])), grey),
            ('glide_refelection', ((1, self.point[1][1]), (self.info[0] // 2, self.point[1][1] * 2)), grey),

            #  旋转中心点
            ('tri', (self.info[0], self.point[1][1]), red),
            ('tri', (self.info[0] * 2, self.point[1][1]), red),
            ('tri', (self.info[0] // 2, self.point[1][1] * 2), red),
            ('tri', (int(self.info[0] * (3 / 2)), self.point[1][1] * 2), red),
            ('tri', (int(self.info[0] * (3 / 2)), 0), red),
            ('tri', (self.info[0] // 2, 0), red),
            ('tri', (0, self.point[1][1]), red),

        ]
        self.markImg = draw(self.markImg, draw_list)
        func_list = [
            ('over_spread',),
            ('save',),
            ('show',)
        ]
        self.run(func_list=func_list)

    def save(self):
        if self.is_save:
            self.markImg.save(self.save_image_path + '/p3_min.png')
            self.GroundImg.save(self.save_image_path + '/p3_result.png')


class P3m1Line(P3m1):
    def __init__(self, **kwargs):
        super(P3m1Line, self).__init__(**kwargs)

    def generate(self):
        func_list = [
            ('paste_img', (0, 0)),
            ('update_img',),
            ('mirror_angle', 60, (0, 0), None, False),
            ('rotate', -120, (0, 0), None, False),
            ('update_img',),
            ('tb_mirror', (0, 0))
        ]
        self.run(func_list=func_list)
        w, h = self.rectangle_width, self.rectangle_height
        draw_list = [
            # 边框
            ('glide_refelection', ((self.info[0] // 2, 0), (int(self.info[0] * (3 / 2)), 0)), grey),
            ('glide_refelection', ((self.info[0] // 2, 0), (0, self.point[1][1])), grey),
            ('glide_refelection', ((1, self.point[1][1]), (self.info[0] // 2, self.point[1][1] * 2)), grey),

            # 镜像
            ('refelection', ((w // 4, 0), (int(w * (3 / 4)), h)), blue),
            ('refelection', ((int(w * (3 / 4)), 0), (w // 4, h)), blue),
            ('refelection', ((0, h // 2), (w, h // 2)), blue),

            ('refelection', ((self.info[0] // 2, 0), (int(self.info[0] * (3 / 2)), 0)), blue),
            ('refelection', ((self.info[0] // 2, 0), (0, h // 2)), blue),
            ('refelection', ((self.info[0] // 2, h), (0, h // 2)), blue),

            # glide_refelection
            ('glide_refelection', ((0, h // 4), (w, h // 4)), green),
            ('glide_refelection', ((0, int(h * (3 / 4))), (w, int(h * (3 / 4)))), green),

            ('glide_refelection', ((w // 2, 0), (w // 8, int(h * (3 / 4)))), green),
            ('glide_refelection', ((w // 2, 0), (int(w * (7 / 8)), int(h * (3 / 4)))), green),

            ('glide_refelection', ((w // 8, h // 4), (w // 2, h)), green),
            ('glide_refelection', ((int(w * (7 / 8)), h // 4), (w // 2, h)), green),

            #  旋转中心点
            ('tri', (self.info[0], self.point[1][1]), red),
            ('tri', (self.info[0] * 2, self.point[1][1]), red),
            ('tri', (self.info[0] // 2, self.point[1][1] * 2), red),
            ('tri', (int(self.info[0] * (3 / 2)), self.point[1][1] * 2), red),
            ('tri', (int(self.info[0] * (3 / 2)), 0), red),
            ('tri', (self.info[0] // 2, 0), red),
            ('tri', (0, self.point[1][1]), red),
        ]
        self.markImg = draw(self.markImg, draw_list)
        func_list = [
            ('over_spread',),
            ('save',),
            ('show',)
        ]
        self.run(func_list=func_list)

    def save(self):
        if self.is_save:
            self.markImg.save(self.save_image_path + '/p3m1_min.png')
            self.GroundImg.save(self.save_image_path + '/p3m1_result.png')


class P31mLine(P31m):
    def __init__(self, **kwargs):
        super(P31mLine, self).__init__(**kwargs)

    def generate(self):
        func_list = [('paste_img', (0, self.info[1])),
                     ('update_img',),
                     ('rotate', -120, (0, 0), (self.point[0][0], self.info[1]), False),
                     ('rotate', 120, (0, 0), (self.point[0][0], self.info[1]), False),
                     ('update_img',),
                     ('mirror_angle', 60, (0, 0), None, False),
                     ('paste_img', (self.info[0], 0)),
                     ('update_img',),
                     ('tb_mirror', (0, 0))
                     ]
        self.run(func_list=func_list)

        w, h = self.rectangle_width, self.rectangle_height
        draw_list = [
            # 镜像
            ('refelection', ((w // 4, 0), (int(w * (3 / 4)), h)), blue),
            ('refelection', ((int(w * (3 / 4)), 0), (w // 4, h)), blue),
            ('refelection', ((0, h // 2), (w, h // 2)), blue),

            ('refelection', ((self.info[0] // 2, 0), (int(self.info[0] * (3 / 2)), 0)), blue),
            ('refelection', ((self.info[0] // 2, 0), (0, h // 2)), blue),
            ('refelection', ((self.info[0] // 2, h), (0, h // 2)), blue),

            # glide_refelection
            ('glide_refelection', ((0, h // 4), (w, h // 4)), green),
            ('glide_refelection', ((0, int(h * (3 / 4))), (w, int(h * (3 / 4)))), green),

            ('glide_refelection', ((w // 2, 0), (w // 8, int(h * (3 / 4)))), green),
            ('glide_refelection', ((w // 2, 0), (int(w * (7 / 8)), int(h * (3 / 4)))), green),

            ('glide_refelection', ((w // 8, h // 4), (w // 2, h)), green),
            ('glide_refelection', ((int(w * (7 / 8)), h // 4), (w // 2, h)), green),

            #  旋转中心点
            ('tri', (self.info[0], self.point[1][1]), red),
            ('tri', (self.info[0] // 2, self.point[1][1] * 2), red),
            ('tri', (int(self.info[0] * (3 / 2)), self.point[1][1] * 2), red),
            ('tri', (int(self.info[0] * (3 / 2)), 0), red),
            ('tri', (self.info[0] // 2, 0), red),
            ('tri', (0, self.point[1][1] + self.info[1]), red),
            ('tri', (self.info[0] // 2, (self.point[1][1] + self.info[1]) * 2), red),
            ('tri', (int(self.info[0] * (3 / 2)), (self.point[1][1] + self.info[1]) * 2), red),
            ('tri', (self.info[0] * 2, self.point[1][1] + self.info[1]), red),
            ('tri', (self.info[0], self.point[1][1] + self.info[1]), red),

            ('tri', (self.info[0], self.point[1][1] + self.info[1] * 2), red),
            ('tri', (self.info[0] // 2, self.point[1][1] * 2 + self.info[1]), red),
            ('tri', (int(self.info[0] * (3 / 2)), self.point[1][1] * 2 + self.info[1]), red),

        ]
        self.markImg = draw(self.markImg, draw_list)
        func_list = [
            ('over_spread',),
            ('save',),
            ('show',)
        ]
        self.run(func_list=func_list)

    def save(self):
        if self.is_save:
            self.markImg.save(self.save_image_path + '/p31m_min.png')
            self.GroundImg.save(self.save_image_path + '/p31m_result.png')


class P6Line(P6):
    def __init__(self, **kwargs):
        super(P6Line, self).__init__(**kwargs)

    def generate(self):
        func_list = [('paste_img', (0, 0)),
                     ('update_img',),
                     ('rotate', 60, (0, 0), None, False),
                     ('rotate', 120, (0, 0), None, False),
                     ('rotate', -60, (0, 0), None, False),
                     ('rotate', -120, (0, 0), None, False),
                     ('rotate', 180, (0, 0), None, False)
                     ]
        self.run(func_list=func_list)

        w, h = self.rectangle_width, self.rectangle_height
        draw_list = [
            # 边框
            ('glide_refelection', ((self.info[0] // 2, 0), (int(self.info[0] * (3 / 2)), 0)), grey),
            ('glide_refelection', ((self.info[0] // 2, 0), (0, self.point[1][1])), grey),
            ('glide_refelection', ((1, self.point[1][1]), (self.info[0] // 2, self.point[1][1] * 2)), grey),

            #  旋转中心点
            ('hex', (self.info[0], self.point[1][1]), blue),
            ('tri', (self.info[0] * 2, self.point[1][1]), red),
            ('tri', (self.info[0] // 2, self.point[1][1] * 2), red),
            ('tri', (int(self.info[0] * (3 / 2)), self.point[1][1] * 2), red),
            ('tri', (int(self.info[0] * (3 / 2)), 0), red),
            ('tri', (self.info[0] // 2, 0), red),
            ('tri', (0, self.point[1][1]), red),

            ('rhomb', (w // 2, 0), green),
            ('rhomb', (w // 8, h // 4), green),
            ('rhomb', (w // 8, int(h * (3 / 4))), green),

            ('rhomb', (w // 2, h), green),
            ('rhomb', (int(w * (7 / 8)), int(h * (3 / 4))), green),
            ('rhomb', (int(w * (7 / 8)), h // 4), green),

        ]
        self.markImg = draw(self.markImg, draw_list)
        func_list = [
            ('over_spread',),
            ('show',),
            ('save',)
        ]
        self.run(func_list=func_list)

    def save(self):
        if self.is_save:
            self.markImg.save(self.save_image_path + '/p6_min.png')
            self.GroundImg.save(self.save_image_path + '/p6_result.png')


class P6mLine(P6m):
    def __init__(self, **kwargs):
        super(P6mLine, self).__init__(**kwargs)

    def generate(self):
        self.info[0] = 2 * self.info[0]
        func_list = [('paste_img', (self.point[2][0], self.point[2][1])),
                     ('lr_mirror', (2 * self.point[2][0], self.point[2][1])),
                     ('update_img',),
                     ('rotate', 60, (0, 0), None, False),
                     ('rotate', 120, (0, 0), None, False),
                     ('rotate', -60, (0, 0), None, False),
                     ('rotate', -120, (0, 0), None, False),
                     ('rotate', 180, (0, 0), None, False)
                     ]
        self.run(func_list=func_list)

        w, h = self.rectangle_width, self.rectangle_height
        draw_list = [
            # 边框
            ('glide_refelection', ((w // 4, 0), (int(w * (3 / 4)), 0)), grey),
            ('glide_refelection', ((w // 4, 0), (0, h // 2)), grey),
            ('glide_refelection', ((0, h // 2 - 1), (w // 4, h)), grey),

            # 镜像
            ('refelection', ((w // 4, 0), (int(w * (3 / 4)), h)), blue),
            ('refelection', ((int(w * (3 / 4)), 0), (w // 4, h)), blue),
            ('refelection', ((0, h // 2), (w, h // 2)), blue),

            ('refelection', ((w // 2, 0), (w // 2, h)), blue),
            ('refelection', ((w // 8, h // 4), (int(w * (7 / 8)), int(h * (3 / 4)))), blue),
            ('refelection', ((int(w * (7 / 8)), h // 4), (h // 8, int(h * (3 / 4)))), blue),

            # 旋转中心点
            ('rhomb', (w // 2, 0), green),
            ('rhomb', (w // 8, h // 4), green),
            ('rhomb', (w // 8, int(h * (3 / 4))), green),

            ('rhomb', (w // 2, h), green),
            ('rhomb', (int(w * (7 / 8)), int(h * (3 / 4))), green),
            ('rhomb', (int(w * (7 / 8)), h // 4), green),

            ('hex', (w // 2, h // 2), red),

            ('tri', (w // 4, 0), orange),
            ('tri', (int(w * (3 / 4)), 0), orange),
            ('tri', (0, h // 2), orange),
            ('tri', (w, h // 2), orange),
            ('tri', (w // 4, h), orange),
            ('tri', (int(w * (3 / 4)), h), orange),

        ]
        self.markImg = draw(self.markImg, draw_list)

        func_list = [
            ('over_spread',),
            ('save',),
            ('show',)
        ]
        self.run(func_list=func_list)

    def save(self):
        if self.is_save:
            self.markImg.save(self.save_image_path + '/p6m_min.png')
            self.GroundImg.save(self.save_image_path + '/p6m_result.png')
