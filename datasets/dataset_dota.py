import os
import cv2
import numpy as np
from .base import BaseDataset
from .DOTA_devkit.ResultMerge_multi_process import mergebypoly

def idfclock(box):
    idx = (box[1][0]-box[0][0])*(box[2][1]-box[1][1])-(box[1][1]-box[0][1])*(box[2][0]-box[1][0])
    if idx<0:
        new_box = [box[0],box[3],box[2],box[1]]
    else:
        new_box = box
    return new_box


class DOTA(BaseDataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=None):
        super(DOTA, self).__init__(data_dir, phase, input_h, input_w, down_ratio)
        # 类别名称
        #self.category = ['baseball-diamond', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'tennis-court','basketball-court', 'roundabout']
        #self.category = ['plane', 'small-vehicle', 'large-vehicle','ship', 'tennis-court', 'storage-tank',"helicopter"]
        self.category = ['1', '2', '3', '4', '5']
        self.color_pans = [(204,78,210), (0,192,255), (0,131,0), (240,176,0), (254,100,38), (0,0,255), (182,117,46),
                           (185,60,129), (204,153,255), (80,208,146), (0,0,204), (17,90,197), (0,255,255), (102,255,102),
                           (255,255,0)]
        self.num_classes = len(self.category)                         # 类别
        self.cat_ids = {cat:i for i,cat in enumerate(self.category)}  # 类别与id的映射
        self.img_ids = self.load_img_ids()                            # 加载图片id
        self.image_path = os.path.join(data_dir, 'images')            # img
        self.label_path = os.path.join(data_dir, 'labelTxt')          # label

    # 加载图片索引
    def load_img_ids(self):
        if self.phase == 'train':
            image_set_index_file = os.path.join(self.data_dir, 'trainval.txt')   # image_idx
        else:
            image_set_index_file = os.path.join(self.data_dir, self.phase + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file, 'r') as f:
            lines = f.readlines()    # 读所有内容
        image_lists = [line.strip() for line in lines]   # 根据空格分
        return image_lists

    # 加载图片
    def load_image(self, index):
        img_id = self.img_ids[index]
        imgFile = os.path.join(self.image_path, img_id+'.tif')   # .png  .tif
        assert os.path.exists(imgFile), 'image {} not existed'.format(imgFile)
        img = cv2.imread(imgFile)
        return img

    # 获取标注文件路径
    def load_annoFolder(self, img_id):
        return os.path.join(self.label_path, img_id+'.txt')

    # 加载标注文件
    def load_annotation(self, index):
        image = self.load_image(index)
        h,w,c = image.shape
        valid_pts = []
        valid_cat = []
        valid_dif = []
        with open(self.load_annoFolder(self.img_ids[index]), 'r') as f:
            for i, line in enumerate(f.readlines()):
                obj = line.strip('\n').strip(' ').split(' ')  # list object
                if len(obj)>8:
                    x1 = min(max(float(obj[1]), 0), w - 1)
                    y1 = min(max(float(obj[2]), 0), h - 1)
                    x2 = min(max(float(obj[3]), 0), w - 1)
                    y2 = min(max(float(obj[4]), 0), h - 1)
                    x3 = min(max(float(obj[5]), 0), w - 1)
                    y3 = min(max(float(obj[6]), 0), h - 1)
                    x4 = min(max(float(obj[7]), 0), w - 1)
                    y4 = min(max(float(obj[8]), 0), h - 1)
                    # 过滤掉小目标（注意啊？？？？？？？？？？？？？？？？？？）
                    xmin = max(min(x1, x2, x3, x4), 0)
                    xmax = max(x1, x2, x3, x4)
                    ymin = max(min(y1, y2, y3, y4), 0)
                    ymax = max(y1, y2, y3, y4)
                    if ((xmax - xmin) > 10) and ((ymax - ymin) > 10):
                        yan_box = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                        yan_box = idfclock(yan_box)
                        valid_pts.append(yan_box)
                        valid_cat.append(self.cat_ids[obj[0]])
                        valid_dif.append(0)
        f.close()
        annotation = {}
        annotation['pts'] = np.asarray(valid_pts, np.float32)  # bbox
        annotation['cat'] = np.asarray(valid_cat, np.int32)    # 分类
        annotation['dif'] = np.asarray(valid_dif, np.int32)    # 难易程度
        return annotation

    # 结果合并函数
    def merge_crop_image_results(self, result_path, merge_path):
        mergebypoly(result_path, merge_path)
