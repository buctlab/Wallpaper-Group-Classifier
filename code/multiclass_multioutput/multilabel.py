import threading
import copy
import time
import numpy as np
import torch
import random
from multiclass_multioutput.prediction import Prediction
from multiclass_multioutput.picture_loader import MultiLoader, MirrorLoader
from net.cnn import CNN5
from rotate.plot_figure import plot_CM
import csv


def draw_conf_matrix(pre_result, classes_name, data_class_name, title, normalize=False, save_folder='conf_matrix',
                     **kwargs):
    """
    Args:
        pre_result:
        classes_name: list of the actual data_label per data:x_label
        data_class_name: y_label
        title:

    Returns:

    """
    # 创建一个空矩阵存储混淆矩阵
    class_list = np.concatenate(classes_name)
    y_ticks = kwargs.get('y_ticks', classes_name)
    # conf_matrix = torch.zeros(len(classes_name), 17)
    conf_matrix = torch.zeros(17, len(y_ticks))
    for i in range(len(pre_result)):
        index = np.where(class_list == data_class_name[i])
        # conf_matrix[pre_result[i], index] += 1
        conf_matrix[index, pre_result[i]] += 1

    present = time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime())
    save_path = '{}/{}_{}_{}_{}.png'.format(save_folder, title, len(classes_name), len(data_class_name), present)
    print('fig_save_path:', save_path)

    f = open('{}/data_{}_{}.csv'.format(save_folder, title, present), 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerows(conf_matrix.numpy().tolist())
    f.close()

    # plot_CM(conf_matrix.numpy(), class_list, classes_name, save_path=save_path, normalize=False,
    # title='prediction matrix')
    plot_CM(conf_matrix.numpy(), x_ticks=y_ticks, y_ticks=class_list, save_path=save_path, normalize=normalize,
            title=" ")  # title='prediction matrix'


class MyThread(threading.Thread):
    def __init__(self, target, args, kwargs, name=''):
        threading.Thread.__init__(self)
        self.name = name
        self.target = target
        self.args = args
        self.kwargs = kwargs
        print('start----------------------------------------------Thread_name:【', self.getName(), '】')
        self.result = self.target(*self.args, **self.kwargs)
        print('end----------------------------------------------Thread_name:【', self.getName(), '】')

    def get_result(self):
        try:
            return self.result
        except:
            return None


class MultiClassWrapper(object):
    def __init__(self, base_classifier, classes_name, label_matrix, folder_path, pthfile):
        """
        :param base_classifier: 实例化后的分类器
        :param mode: 'ovr'表示one-vs-rest方式,'ovo'表示one-vs-one方式
        """
        self.base_classifier = base_classifier
        self.classifiers = []
        self.classifier_num = 5
        self.classes_name = classes_name
        self.label_matrix = label_matrix
        # self.class_dict = dict_zip(classes_name, label_matrix)

        self.folder_path = folder_path
        self.pthfile = pthfile

    @staticmethod
    def predict_proba_base_classifier(load, prediction, base_classifier, pthfile):
        return prediction.prediction_run17(base_classifier, load, pth_file=pthfile)

    def predict_proba(self, class_num, per_class_num, img_size, batch_size, data_load_kind, **kwargs):
        for cls in range(self.classifier_num):
            # for cls in range(1):
            self.classifiers.append(copy.deepcopy(self.base_classifier))

        tasks = []
        probas = []

        # 预测（数据）
        prediction = Prediction(img_size=img_size, classes=self.classes_name, folder_path=self.folder_path,
                                class_num=class_num, batch_size=batch_size,
                                per_class_num=per_class_num)
        data_load, data_class_name = prediction.data17(MultiLoader, data_load_kind=data_load_kind)

        # 并行
        for cls in range(len(self.classifiers)):
            # print(cls, self.pthfile[cls])
            task = MyThread(target=self.predict_proba_base_classifier,
                            args=(
                                data_load, prediction, self.classifiers[cls], self.pthfile[cls]),
                            kwargs=kwargs, name=str(cls))
            task.start()
            tasks.append(task)

        for task in tasks:
            task.join()
        for task in tasks:
            probas.append(task.get_result())
        # 统计概率
        # total_probas = np.concatenate(probas, axis=1)  # 数组拼接
        total_probas = np.array(probas).swapaxes(0, 1)

        return total_probas, data_load, data_class_name

    def predict(self, class_num, per_class_num, img_size, batch_size, data_load_kind='normal'):
        probas, data_load, data_class_name = self.predict_proba(class_num, per_class_num, img_size,
                                                                batch_size=batch_size,
                                                                data_load_kind=data_load_kind)
        # label = [self.classes_name.index(i) for i in data_class_name]
        return probas, data_class_name

    def distance(self, probas, mode, pre_shape_result=None, candidate_label_matrix=None):
        # probas--self.label_matrix
        # pre_shape_result--candidate_label_matrix
        if mode == 'norm':
            # distance = hamming_distance(probas, self.label_matrix)
            distance = error_distance(probas, self.label_matrix)
        elif mode == 'candidate':
            # 候选列表
            distance = hamming_distance_upgrade(probas, self.label_matrix, pre_shape_result, candidate_label_matrix)
        elif mode == 'two':
            distance = hamming_distance2(probas, self.label_matrix, pre_shape_result, candidate_label_matrix)
        else:
            return
        pre_result = random_min(distance)  # 随机取最小值
        # pre_result = np.argmin(distance, axis=1)  # 直接按位置最小值

        # print('distance:', distance)
        # print('pre:', pre_result)

        return pre_result


def random_min(data):
    # 最小值有重复, 随机取最小值
    pre_result = []

    min = np.min(data, axis=1)

    for i in range(len(min)):
        result = np.where(data[i] == min[i])

        random.shuffle(result[0])
        pre_result.append(result[0][0])
    # pre_result1 = np.argmin(data, axis=1)

    return pre_result


def hamming_distance(probas, label_matrix):
    # probas----label_matrix
    # pre_result = []
    distance = []
    for x in probas:
        base_distance = []
        for y in label_matrix:
            i = temp = 0
            while i < len(x):
                if not x[i] == y[i]:
                    temp = temp + 1
                i += 1
            base_distance.append(temp)
        distance.append(base_distance)
    return distance


def hamming_distance2(probas, label_matrix, pre_shape_result, candidate_label_matrix):
    # probas----label_matrix, candidate_label_matrix（俩矩阵选用不同的方法计算距离相加）
    # pre_result = []
    distance = []
    i = 0
    for x in probas:
        # print('distance:\n', x)
        base_distance = []
        j = 0
        for y in label_matrix:
            k = temp = 0
            while k < len(x):
                if not x[k] == y[k]:
                    temp += 1
                k += 1
            # shape距离
            if candidate_label_matrix[j, pre_shape_result[i]]:
                temp -= 1
            base_distance.append(temp)
            j += 1
        i += 1
        distance.append(base_distance)
    return distance


def hamming_distance_upgrade(probas, label_matrix, pre_shape_result, candidate_label_matrix):
    # 加入基本块候选类别
    # pre_shape_result(结果)————>candidate_label_matrix(筛选)
    distance = []
    i = 0
    for x in probas:
        candidate_index = np.nonzero(candidate_label_matrix[:, pre_shape_result[i]])[0]
        # print('candidate_index:', candidate_index)
        base_distance = [100] * len(label_matrix)
        for j in candidate_index:
            y = label_matrix[j]
            k = temp = 0
            while k < len(x):
                if not x[k] == y[k]:
                    temp = temp + 1
                k += 1
            # base_distance.append(temp)
            base_distance[j] = temp
        distance.append(base_distance)
        i += 1
    return distance


def error_distance(differ, label_matrix):
    distance = []
    for x in differ:
        base_distance = []
        for label in label_matrix:
            count = 0 if len(x) == 5 else 1 - x[5 + label[5] * 2 + label[6]]
            for i in range(5):
                count += abs(label[i] - x[i])
            base_distance.append(count)
        distance.append(base_distance)
    return distance


def binary_add(a):
    temp = 1
    # print('a:', a)
    for i in range(len(a)):
        a[i] += temp
        temp = a[i] // 2
        # print(temp)
        if temp == 0:
            return a
        else:
            a[i] = 0


def exhaustion(label_matrix):
    l = len(label_matrix)
    w = 2 ** l
    base_data = np.zeros(l, dtype=int)
    data = []
    data.append(copy.deepcopy(base_data))
    # print(data)
    for i in range(1, w):
        data.append(copy.deepcopy(binary_add(base_data)))
    distance = hamming_distance(data, label_matrix)


def random_max(data):
    pre_result = []
    max_d = np.max(data, axis=1)
    for i in range(len(max_d)):
        result = np.where(data[i] == max_d[i])
        random.shuffle(result[0])
        pre_result.append(result[0][0])
    return pre_result


def mirror_label(rotate_pth_files, reflection_pth_file, image_folder_path='../data', save_folder='output', batch_size=20,
                 per_class_num=1000, img_size=224, ):
    """
    rotate_pth_files: five .pkl rotate file
    reflection_pth_file: one .pkl reflection file

    image_folder_path: 数据地址
    save_folder: 结果存储地址

    batch_size: batch size
    per_class_num: 每个类别的数据量
    img_size: 输入训练模型的图片大小
    """
    class_num = 17

    classes_name = [['p1'], ['pm'], ['pg'], ['cm'],
                    ['p2'], ['pmm'], ['pmg'], ['pgg', 'cmm'],
                    ['p4'], ['p4m', 'p4g'],
                    ['p3'], ['p3m1', 'p31m'],
                    ['p6'], ['p6m']]

    label_matrix = [[1, 0, 0, 0, 0, 0, 0],  # p1
                    [1, 0, 0, 0, 0, 1, 0],  # pm
                    [1, 0, 0, 0, 0, 0, 1],  # pg
                    [1, 0, 0, 0, 0, 1, 1],  # cm

                    [0, 1, 0, 0, 0, 0, 0],  # p2
                    [0, 1, 0, 0, 0, 1, 0],  # pmm
                    [0, 1, 0, 0, 0, 0, 1],  # pgg
                    [0, 1, 0, 0, 0, 1, 1],  # pmg,cmm

                    [0, 1, 1, 0, 0, 0, 0],  # p4
                    [0, 1, 1, 0, 0, 1, 0],  # p4m,p4g

                    [0, 0, 0, 1, 0, 0, 0],  # p3
                    [0, 0, 0, 1, 0, 1, 0],  # p3m1,p31m

                    [0, 1, 0, 1, 1, 0, 0],  # p6
                    [0, 1, 0, 1, 1, 1, 0]]  # p6m

    # ____________________________________
    # (角度
    Net = CNN5(img_size, 2)

    ovr = MultiClassWrapper(Net, classes_name, label_matrix, image_folder_path, rotate_pth_files)
    probas, data_class_name = ovr.predict(class_num=class_num, per_class_num=per_class_num, img_size=img_size,
                                          batch_size=batch_size)
    # 角度训练距离计算，混淆矩阵
    pre_result = ovr.distance(probas, mode='norm')
    draw_conf_matrix(pre_result, classes_name, data_class_name, title='rotate')

    # end)
    # ____________________________________
    # (mirror_glid
    prediction = Prediction(img_size=img_size, classes=classes_name, folder_path=image_folder_path,
                            class_num=class_num, batch_size=batch_size,
                            per_class_num=per_class_num)
    data_load, data_class_name = prediction.data17(MirrorLoader)  # 不能用shuffle，确保所有预测里图片顺序一致
    mirror_glid_classifier = CNN5(img_size, 4)
    pre_mirror_result = prediction.prediction_run17(mirror_glid_classifier, data_load,
                                                    pth_file=reflection_pth_file)
    pre_result = random_max(pre_mirror_result)
    draw_conf_matrix(pre_result, classes_name, data_class_name, title='reflection',
                     y_ticks=['FF', 'FT', 'TF', 'TT'], save_folder=save_folder)
    draw_conf_matrix(pre_result, classes_name, data_class_name, title='reflection',
                     y_ticks=['FF', 'FT', 'TF', 'TT'], normalize=True, save_folder=save_folder)

    # end)
    # _____________________________________

    # 分类结果综合
    probas = np.hstack((probas, np.array(pre_mirror_result)))
    pre_result = ovr.distance(probas, mode='norm')  # 汉明距离分类结果
    # 混淆矩阵
    draw_conf_matrix(pre_result, classes_name, data_class_name, title='ro_ref', save_folder=save_folder)
    draw_conf_matrix(pre_result, classes_name, data_class_name, title='ro_ref', normalize=True, save_folder=save_folder)


if __name__ == '__main__':
    pth_path = '../rotate/output/info/2022-10-16/'
    pth_files = [pth_path + '2022-10-16_13.26.55/epoch_11_20_2022-10-16_13.26.55.pkl',
                 pth_path + '2022-10-16_17.42.14/epoch_14_20_2022-10-16_17.42.14.pkl',
                 pth_path + '2022-10-16_23.02.18/epoch_10_20_2022-10-16_23.02.18.pkl',
                 pth_path + '2022-10-17_03.02.36/epoch_18_20_2022-10-17_03.02.36.pkl',
                 pth_path + '2022-10-17_07.44.44/epoch_13_20_2022-10-17_07.44.44.pkl']
    pth_file = '../reflection/output/2022-10-17/2022-10-17_23.30.58/epoch_16_20_2022-10-17_23.30.58.pkl'

    mirror_label(pth_files, pth_file, per_class_num=1000)
