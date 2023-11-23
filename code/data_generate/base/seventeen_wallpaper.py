from base.cell import Cell


class P1(Cell):
    def __init__(self, **kwargs):
        point, info = kwargs['point'], kwargs['info']
        w, h = self.rectangle_init(point, info)
        kwargs.update({'kind': info[-1], 'rectangle_width': w, 'rectangle_height': h})
        super(P1, self).__init__(**kwargs)

    @staticmethod
    def rectangle_init(point, info):
        kind = info[-1]
        height = point[2][1]
        if kind == 5:
            width = point[4][0]
        else:
            width = point[3][0]
        return width, height

    def generate(self):
        func_list = [('paste_img', (0, 0)),
                     ('over_spread', self.point[0]),
                     ('save',),
                     ('show',)
                     ]
        self.run(func_list=func_list)


class P2(Cell):
    def __init__(self, **kwargs):
        point, info = kwargs['point'], kwargs['info']
        kind, w, h = self.rectangle_init(point, info)
        kwargs.update({'kind': kind, 'rectangle_width': w, 'rectangle_height': h})
        super(P2, self).__init__(**kwargs)

    @staticmethod
    def rectangle_init(point, info):
        kind = info[-1]
        if kind == 1 or kind == 2:
            h = point[1][1] * 2
            w = point[2][0]
        elif kind == 3:
            h = 2 * point[1][1]
            w = point[2][0] + point[0][0] * 2
            kind = 3
        elif kind == 6:
            w = point[2][0]
            h = point[1][1] * 2
            kind = 4  # 按照菱形拼
        else:
            w = h = 0
        return kind, w, h

    def generate(self):
        func_list = [('paste_img', (0, 0)),
                     ('update_img',),
                     ('rotate', -180, (0, 0)),
                     ('over_spread',),
                     ('save',),
                     ('show',)
                     ]
        if self.kind == 3:
            func_list[0] = ('paste_img', (0, self.rectangle_height // 2))
            func_list[3] = ('over_spread', (self.point[0][0] * 2, 0))
        self.run(func_list=func_list)


class P2Hexagonal(Cell):
    def __init__(self, **kwargs):
        point = kwargs['point']
        kind, w, h = self.rectangle_init(point)
        kwargs.update({'kind': kind, 'rectangle_width': w, 'rectangle_height': h})
        super(P2Hexagonal, self).__init__(**kwargs)

    @staticmethod
    def rectangle_init(point):
        # 必须传入等边三角形
        w = point[0][0] + point[2][0]
        h = point[1][1]
        kind = 3  # 按照平行四边形拼
        return kind, w, h

    def generate(self):
        func_list = [('paste_img', (0, 0)),
                     ('update_img',),
                     ('rotate', -180, (0, 0)),
                     ('over_spread', (self.point[0][0], 0)),
                     ('save',),
                     ('show',)
                     ]
        self.run(func_list=func_list)


class Pm(Cell):
    def __init__(self, **kwargs):
        point, info = kwargs['point'], kwargs['info']
        w, h = self.rectangle_init(point, info)
        kwargs.update({'kind': info[-1], 'rectangle_width': w, 'rectangle_height': h})
        super(Pm, self).__init__(**kwargs)

    @staticmethod
    def rectangle_init(point, info):
        kind = info[-1]
        if kind == 1:
            w = point[1][1] * 2
            h = point[1][1]
        elif kind == 2:
            w = point[2][0]
            h = point[1][1] * 2
        else:
            raise ValueError("kind should between 1 and 2")
        return w, h

    def generate(self):
        func_list = [('paste_img', (0, 0)),
                     ('update_img',),
                     ('lr_mirror', (0, 0)),
                     ('over_spread',),
                     ('save',),
                     ('show',)
                     ]
        if self.info[-1] == 2:
            func_list[2] = ('tb_mirror', (0, 0))
        else:
            func_list = func_list
        self.run(func_list=func_list)


class Pg(Cell):
    def __init__(self, **kwargs):
        point, info = kwargs['point'], kwargs['info']
        w, h = self.rectangle_init(point)
        kwargs.update({'kind': info[-1], 'rectangle_width': w, 'rectangle_height': h})
        super(Pg, self).__init__(**kwargs)

    @staticmethod
    def rectangle_init(point):
        w = point[2][0]
        h = point[1][1] * 2
        return w, h

    def generate(self):
        func_list = [('paste_img', (0, 0)),
                     ('update_img',),
                     ('lr_mirror', (0, self.point[1][1])),
                     ('over_spread',),
                     ('save',),
                     ('show',)
                     ]
        self.run(func_list=func_list)


class Cm(Cell):
    def __init__(self, **kwargs):
        point = kwargs['point']
        kind, w, h = self.rectangle_init(point)
        kwargs.update({'kind': kind, 'rectangle_width': w, 'rectangle_height': h})
        super(Cm, self).__init__(**kwargs)

    @staticmethod
    def rectangle_init(point):
        w_w = point[2][0]
        h_h = point[1][1] * 2
        kind = 4
        return kind, w_w, h_h

    def generate(self):
        func_list = [('paste_img', (0, 0)),
                     ('update_img',),
                     ('tb_mirror', (0, 0)),
                     ('over_spread',),
                     ('save',),
                     ('show',)
                     ]
        self.run(func_list=func_list)


class Pmm(Cell):
    def __init__(self, **kwargs):
        point = kwargs['point']
        info = kwargs['info']
        w, h = self.rectangle_init(point, info)
        kwargs.update({'kind': info[-1], 'rectangle_width': w, 'rectangle_height': h})
        super(Pmm, self).__init__(**kwargs)

    @staticmethod
    def rectangle_init(point, info):
        w_w = point[2][0] * 2
        h_h = point[1][1] * 2
        return w_w, h_h

    def generate(self):
        func_list = [('paste_img', (0, 0)),
                     ('update_img',),
                     ('lr_mirror', (0, 0)),
                     ('update_img',),
                     ('tb_mirror', (0, 0)),
                     ('over_spread',),
                     ('save',),
                     ('show',)
                     ]
        self.run(func_list=func_list)


class Pmg(Cell):
    def __init__(self, **kwargs):
        point = kwargs['point']
        kind, w, h = self.rectangle_init(point)
        kwargs.update({'kind': kind, 'rectangle_width': w, 'rectangle_height': h})
        super(Pmg, self).__init__(**kwargs)

    @staticmethod
    def rectangle_init(point):
        w_w = point[3][0] * 2
        h_h = point[1][1] * 2
        kind = 2
        return kind, w_w, h_h

    def generate(self):
        func_list = [('paste_img', (0, 0)),
                     ('rotate', 180, (self.info[0], 0)),
                     ('update_img',),
                     ('tb_mirror', (0, 0)),
                     ('over_spread',),
                     ('save',),
                     ('show',)
                     ]
        self.run(func_list=func_list)


class Pgg(Cell):
    def __init__(self, **kwargs):
        point, info = kwargs['point'], kwargs['info']
        kind, w, h = self.rectangle_init(point, info)
        kwargs.update({'kind': kind, 'rectangle_width': w, 'rectangle_height': h})
        super(Pgg, self).__init__(**kwargs)

    @staticmethod
    def rectangle_init(point, info):
        kind = info[-1]
        if kind == 6:
            w_w = point[2][0]
            h_h = point[1][1] * 2
            kind = 2  # 按照矩形、正方形拼
        else:
            return
        return kind, w_w, h_h

    def generate(self):
        func_list = [('paste_img', (0, 0)),
                     ('tb_mirror', (self.point[0][0], 0)),
                     ('tb_mirror', (-self.point[0][0], 0)),
                     ('update_img',),
                     ('rotate', 180, (0, 0)),
                     ('over_spread',),
                     ('save',),
                     ('show',)
                     ]
        self.run(func_list=func_list)


class Cmm(Cell):
    def __init__(self, **kwargs):
        self.type = kwargs['type']
        point = kwargs['point']
        kind, w, h = self.rectangle_init(point)
        kwargs.update({'kind': kind, 'rectangle_width': w, 'rectangle_height': h})
        super(Cmm, self).__init__(**kwargs)

    def rectangle_init(self, point):
        if self.type == 'rhombic':
            kind = 4
            w_w = 2 * point[2][0]
            h_h = 2 * point[2][1]
            return kind, w_w, h_h
        elif self.type == 'square':
            w_w = h_h = point[2][0]
            kind = 1  # 按正方形
            return kind, w_w, h_h
        else:
            raise ValueError('unknown type : rhombic or square only')

    def generate(self):
        if self.type == 'rhombic':
            func_list = [('paste_img', (0, 0)),
                         ('update_img',),
                         ('tb_mirror', (0, 0)),
                         ('update_img',),
                         ('lr_mirror', (0, 0)),
                         ('over_spread',),
                         ('save',),
                         ('show',)
                         ]
        elif self.type == 'square':
            func_list = [('paste_img', (0, int(self.point[2][0] // 2))),
                         ('update_img',),
                         ('mirror_angle', -90, (0, 0)),
                         ('mirror_angle', 90, (0, 0)),
                         ('rotate', -180, (0, 0)),
                         ('over_spread',),
                         ('save',),
                         ('show',)
                         ]
        else:
            raise ValueError('unknown type : rhombic or square only')
        self.run(func_list=func_list)


# class CmmSquare(Cell):
#     def __init__(self, **kwargs):
#         point = kwargs['point']
#         kind, w, h = self.rectangle_init(point)
#         super(CmmSquare, self).__init__(kind=kind, rectangle_width=w, rectangle_height=h, **kwargs)
#
#     def rectangle_init(self, point):
#         w_w = h_h = point[2][0]
#         kind = 1  # 按正方形
#         return kind, w_w, h_h
#
#     def generate(self):
#         func_list = [('paste_img', (0, int(self.point[2][0] // 2))),
#                      ('update_img',),
#                      ('mirror_angle', -90, (0, 0)),
#                      ('mirror_angle', 90, (0, 0)),
#                      ('rotate', -180, (0, 0)),
#                      ('over_spread',),
#                      ]
#         self.run(func_list=func_list)


class P4(Cell):
    def __init__(self, **kwargs):
        point = kwargs['point']
        info = kwargs['info']
        w, h = self.rectangle_init(point)

        super(P4, self).__init__(kind=info[-1], rectangle_width=w, rectangle_height=h, **kwargs)

    @staticmethod
    def rectangle_init(point):
        w_w = h_h = point[2][0] * 2
        return w_w, h_h

    def generate(self):
        func_list = [('paste_img', (0, 0)),
                     ('update_img',),
                     ('rotate', -90, (0, 0)),
                     ('rotate', 180, (0, 0)),
                     ('rotate', 90, (0, 0)),
                     ('over_spread',),
                     ('save',),
                     ('show',)
                     ]

        self.run(func_list=func_list)


class P4m(Cell):
    def __init__(self, **kwargs):
        point = kwargs['point']
        info = kwargs['info']
        kind, w, h = self.rectangle_init(point, info)
        kwargs.update({'kind': kind, 'rectangle_width': w, 'rectangle_height': h})
        super(P4m, self).__init__(**kwargs)

    @staticmethod
    def rectangle_init(point, info):
        kind = info[-1]
        if kind == 6:  # 必须传入直角三角形
            w_w = point[2][0] * 2
            h_h = point[1][1] * 2
            kind = 1  # 按照正方形拼
        else:
            return
        return kind, w_w, h_h

    def generate(self):

        func_list = [('paste_img', (0, 0)),
                     ('mirror_angle', 90, (0, 0)),
                     ('update_img',),
                     ('tb_mirror', (0, 0)),
                     ('update_img',),
                     ('lr_mirror', (0, 0)),
                     ('over_spread',),
                     ('save',),
                     ('show',)
                     ]

        self.run(func_list=func_list)


class P4g(Cell):
    def __init__(self, **kwargs):
        point = kwargs['point']
        kind, w, h = self.rectangle_init(point)
        kwargs.update({'kind': kind, 'rectangle_width': w, 'rectangle_height': h})
        super(P4g, self).__init__(**kwargs)

    @staticmethod
    def rectangle_init(point):
        kind = 1
        w_w = 2 * point[2][0]
        h_h = 2 * point[2][1]
        return kind, w_w, h_h

    def generate(self):
        func_list = [('paste_img', (0, self.point[2][1])),
                     ('mirror_angle', 90, (0, self.point[2][1])),
                     ('update_img',),
                     ('rotate', -90, (0, 0)),
                     ('rotate', -180, (0, 0)),
                     ('rotate', 90, (0, 0)),
                     ('over_spread',),
                     ('save',),
                     ('show',)
                     ]
        self.run(func_list=func_list)


class P3(Cell):
    def __init__(self, **kwargs):
        point = kwargs['point']
        info = kwargs['info']
        kind, w, h = self.rectangle_init(point, info)
        kwargs.update({'kind': kind, 'rectangle_width': w, 'rectangle_height': h})
        super(P3, self).__init__(**kwargs)

    @staticmethod
    def rectangle_init(point, info):
        # 割出平行四边形（四边相等，夹角为60），经一系列操作后按正六边形拼大画布
        w_w = info[0] * 2
        h_h = point[1][1] * 2
        kind = 5  # 按照正六边形拼

        return kind, w_w, h_h

    def generate(self):
        func_list = [
            ('paste_img', (self.point[0][0], self.point[1][1])),
            ('update_img',),
            ('rotate', -120, (0, 0), None, False),
            ('rotate', 120, (0, 0), None, False),
            ('over_spread',),
            ('save',),
            ('show',)
        ]

        self.run(func_list=func_list)


class P3m1(Cell):
    def __init__(self, **kwargs):
        point = kwargs['point']
        kind, w, h = self.rectangle_init(point)
        kwargs.update({'kind': kind, 'rectangle_width': w, 'rectangle_height': h})
        super(P3m1, self).__init__(**kwargs)

    @staticmethod
    def rectangle_init(point):
        kind = 5
        w_w = 2 * point[2][0]
        h_h = 2 * point[2][1]
        return kind, w_w, h_h

    def generate(self):
        func_list = [('paste_img', (0, 0)),
                     ('update_img',),
                     ('mirror_angle', 60, (0, 0), None, False),
                     ('rotate', -120, (0, 0), None, False),
                     ('update_img',),
                     ('tb_mirror', (0, 0)),
                     ('over_spread',),
                     ('save',),
                     ('show',)
                     ]
        self.run(func_list=func_list)


class P31m(Cell):
    def __init__(self, **kwargs):
        point = kwargs['point']
        info = kwargs['info']
        kind, w, h = self.rectangle_init(info, point)
        kwargs.update({'kind': kind, 'rectangle_width': w, 'rectangle_height': h})
        super(P31m, self).__init__(**kwargs)

    @staticmethod
    def rectangle_init(info, point):
        w_w = info[0] * 2
        h_h = (point[1][1] + info[1]) * 2
        kind = 5  # 按照正六边形拼

        return kind, w_w, h_h

    def generate(self):
        func_list = [('paste_img', (0, self.info[1])),
                     ('update_img',),
                     ('rotate', -120, (0, 0), (self.point[0][0], self.info[1]), False),
                     ('rotate', 120, (0, 0), (self.point[0][0], self.info[1]), False),
                     ('update_img',),
                     ('mirror_angle', 60, (0, 0), None, False),
                     ('paste_img', (self.info[0], 0)),
                     ('update_img',),
                     ('tb_mirror', (0, 0)),
                     ('over_spread',),
                     ('save',),
                     ('show',)
                     ]

        self.run(func_list=func_list)


class P6(Cell):
    def __init__(self, **kwargs):
        point = kwargs['point']
        kind, w, h = self.rectangle_init(point)
        kwargs.update({'kind': kind, 'rectangle_width': w, 'rectangle_height': h})
        super(P6, self).__init__(**kwargs)

    @staticmethod
    def rectangle_init(point):
        kind = 5
        w_w = 2 * point[2][0]
        h_h = 2 * point[2][1]
        return kind, w_w, h_h

    def generate(self):
        func_list = [('paste_img', (0, 0)),
                     ('update_img',),
                     ('rotate', 60, (0, 0), None, False),
                     ('rotate', 120, (0, 0), None, False),
                     ('rotate', -60, (0, 0), None, False),
                     ('rotate', -120, (0, 0), None, False),
                     ('rotate', 180, (0, 0), None, False),
                     ('over_spread',),
                     ('save',),
                     ('show',)
                     ]
        self.run(func_list=func_list)


class P6m(Cell):
    def __init__(self, **kwargs):
        point = kwargs['point']
        kind, w, h = self.rectangle_init(point)
        print('kind',kind)
        kwargs.update({'kind': kind, 'rectangle_width': w, 'rectangle_height': h})
        super(P6m, self).__init__(**kwargs)

    @staticmethod
    def rectangle_init(point):
        kind = 5
        w_w = 4 * point[2][0]
        h_h = 2 * point[2][1]
        return kind, w_w, h_h

    def generate(self):
        self.info[0] = 2 * self.info[0]
        func_list = [('paste_img', (self.point[2][0], self.point[2][1])),
                     ('lr_mirror', (2 * self.point[2][0], self.point[2][1])),
                     ('update_img',),
                     ('rotate', 60, (0, 0), None, False),
                     ('rotate', 120, (0, 0), None, False),
                     ('rotate', -60, (0, 0), None, False),
                     ('rotate', -120, (0, 0), None, False),
                     ('rotate', 180, (0, 0), None, False),
                     ('over_spread',),
                     ('save',),
                     ('show',)
                     ]
        self.run(func_list=func_list)
        self.info[0] = self.info[0] // 2
