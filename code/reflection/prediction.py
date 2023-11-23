from net.cnn import CNN5
from rotate.picture_loader import NoisePictureLoader17
from rotate.prediction import Prediction


def pre(pthfile, folder_path, num=1, img_size=224):
    classes = ['FF', 'FT', 'TF', 'TT']  # 分类类别名称
    # 分类文件名：类别号
    class_dict = {'p1': 0, 'pm': 2, 'pg': 1, 'cm': 3,
                  'p2': 0, 'pmm': 2, 'pmg': 3, 'pgg': 1, 'cmm': 3,
                  'p4': 0, 'p4m': 2, 'p4g': 2,
                  'p3': 0, 'p3m1': 2, 'p31m': 2,
                  'p6': 0, 'p6m': 2}
    title = 'reflection/glide reflection'
    batch_size = 50
    data_load_kind = 'else'  # 数据获取方式1

    net_save = CNN5(img_size, 4)
    pre = Prediction(pth_file=pthfile, model=net_save, img_size=img_size, classes=classes, batch_size=batch_size,
                     title=title, folder_path=folder_path, num=num).prediction_run17(
        NoisePictureLoader17, class_dict, data_load_kind=data_load_kind, is_show=True)
    return pre


if __name__ == "__main__":
    folder_path = ['D:/jupyter_data/images/prediction', 'D:/jupyter_data/images/minor-rotation-prediction']  # 数据路径
    pthfile = 'output/2022-09-21/2022-09-21_18.39.32/epoch_18_20_2022-09-21_18.39.32.pkl'
    pre(pthfile, folder_path, num=3000)  # 图片路径 num=3000
