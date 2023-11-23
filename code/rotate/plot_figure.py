import matplotlib.pyplot as plt
import numpy as np
import csv
import os


def plot_CM(data, save_path, title=" ", normalize=False, **kwargs):
    """
    画混淆矩阵
    Args:
        data: matrix data
        save_path: figure save path(include the figure name)
        title:
        normalize: True:(%)calculate the percentage of rows，False: raw data
    Returns: None
    """

    y_label = 'Actual'
    x_label = 'Predicted'
    if normalize:
        data = data.astype('float') / data.sum(axis=1)[:, np.newaxis]
    # x, y = data.shape
    y, x = data.shape
    x_ticks = kwargs.pop('x_ticks', np.arange(0, x))
    y_ticks = kwargs.pop('y_ticks', np.arange(0, y))
    print('data:({}-{})\n'.format(x, y), data)
    print(x_ticks, y_ticks)

    if x < y:  # 保证长边为横轴
        data = np.transpose(data)
        x, y = y, x
        x_ticks, y_ticks = y_ticks, x_ticks
        x_label, y_label = y_label, x_label

    # with plt.style.context(['science']):  # ['science', 'grid']
    if 1:
        fig, ax = plt.subplots(figsize=(x - 1, y - 1))
        cax = ax.matshow(data, cmap=plt.get_cmap('Blues'))  # 'Blues', 'binary'
        fig.colorbar(cax, pad=0.05, fraction=0.045)
        ax.set_title(title)  # fontsize=10
        ax.set(xlabel=x_label, ylabel=y_label)
        ax.xaxis.set_ticks_position('bottom')

        # 边线不可见
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # 坐标轴标签
        ax.set_xticks(np.arange(0, len(x_ticks)))
        ax.set_yticks(np.arange(0, len(y_ticks)))
        ax.set_xticklabels(x_ticks, rotation=40)  # ,rotation=40
        ax.set_yticklabels(y_ticks)

        # color_threshold = (data.max() - data.min()) / 2
        color_threshold = 0.5
        data_threshold = 0  # data.max() * 0.1
        print(data_threshold)
        for i in range(y):
            for j in range(x):
                if data[i][j] > data_threshold:
                    num = "{:.2%}".format(data[i][j]) if normalize else "{:.0f}".format(data[i][j])
                    plt.text(j, i, num,
                             color='black' if data[i][j] <= color_threshold else '#FFFAFA',
                             va='center', ha='center')
        plt.savefig(save_path, bbox_inches='tight', dpi=500)
        # plt.show()


def plot_csv_data(path, title):
    # 从path获取csv文件数据，画训练数据图像train_test
    f = csv.reader(open(path))
    path = os.path.split(path)[0]
    print('save_path:', path)
    train_loss, test_loss, train_acc, test_acc = list(f)

    train_loss = list(map(float, train_loss[1:]))
    train_acc = list(map(float, train_acc[1:]))
    test_loss = list(map(float, test_loss[1:]))
    test_acc = list(map(float, test_acc[1:]))

    epochs = len(train_acc)
    total_num = len(train_loss)

    mean_train_loss = [np.mean(train_loss[i * (total_num // epochs):(i + 1) * (total_num // epochs)]) for i in
                       range(epochs)]  # 取每个epoch均值

    x = np.linspace(0, epochs - 1, epochs)
    # x_t = np.linspace(0, epochs - epochs / total_num, total_num)

    with plt.style.context(['science', 'grid']):  # 'grid', science, ieee, ggplot
        # if 1:
        fig, ax = plt.subplots(figsize=(6, 4))  # 画布
        ax.grid(linestyle='--')  # 网格
        ax.set_title(title)

        # 4画线
        # ax.plot(x_t, train_loss, label='train loss')
        # pd.Series(train_loss, index=x_t).ewm(span=total_num/epochs).mean().plot(label='mean train loss') # 指数加权α=2/(span+1)
        ax.plot(np.linspace(0, epochs - 1, len(train_loss)), train_loss, label='train loss', marker='')

        # ax.plot(x, mean_train_loss, label='mean train loss', marker='^')
        ax.plot(x, test_loss, label='test loss', marker='H')
        ax.plot(x, train_acc, label='train acc', marker='X')
        ax.plot(x, test_acc, label='test acc', marker='*')

        # 标签
        ax.legend(title='', framealpha=0.85)  # frameon=False)
        # ax.autoscale(tight=True)
        # 坐标轴
        ax.set_xlabel('Epoch')
        ax.xaxis.set_tick_params(rotation=35)  # 坐标轴数字旋转45度
        # start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(0, epochs, 1))  # 标范围
        ax.set_xlim((-0.5, epochs - 0.5))  # 坐标范围
        ax.set_ylim((-0.06, max(1.08, ax.get_ylim()[1])))

        fig.savefig(path + '(5).png', dpi=400, bbox_inches='tight')
        plt.show()


def plot_csv_cm(path, x_ticks, y_ticks, title='Title'):
    data = np.loadtxt(open(path), delimiter=",", skiprows=1)
    print(data.shape, type(data))

    path = os.path.split(path)[0]
    save_path = path + '/plot_csv.png'
    print('save_path:', save_path)
    plot_CM(data, save_path, title, normalize=True, x_ticks=x_ticks, y_ticks=y_ticks)


def plot_csv_cm1(path, ticks, title='Title'):
    data = np.loadtxt(open(path), delimiter=",", skiprows=0)
    data[13] += data[14]
    data = np.delete(data, 14, 0)
    data[10] += data[11]
    data = np.delete(data, 11, 0)
    data[7] += data[8]
    data = np.delete(data, 8, 0)

    ticks[13] = ticks[13] + '\n' + ticks[14]
    ticks = np.delete(ticks, 14, 0)
    ticks[10] = ticks[10] + '\n' + ticks[11]
    ticks = np.delete(ticks, 11, 0)
    ticks[7] = ticks[7] + '\n' + ticks[8]
    ticks = np.delete(ticks, 8, 0)
    # print(data.shape, type(data))

    path = os.path.split(path)[0]
    save_path = path + '/plot_csv.png'
    print('save_path:', save_path)
    plot_CM(data, save_path, title, normalize=True, x_ticks=ticks, y_ticks=ticks)


if __name__ == "__main__":
    plot_csv_data(r'E:\jupyter_tree\group\结果汇总\10.8\policy+mode(LCL)\p31m-p3m1\data.csv', title='')
