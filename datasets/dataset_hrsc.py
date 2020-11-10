from .base import BaseDataset
import os
import cv2
import numpy as np
import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


class HRSC(BaseDataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=None):
        super(HRSC, self).__init__(data_dir, phase, input_h, input_w, down_ratio)
        self.category = ['ship']                     # 类别
        self.num_classes = len(self.category)        # 类别数
        self.cat_ids = {cat:i for i,cat in enumerate(self.category)}   # idx与类别的映射
        self.img_ids = self.load_img_ids()           # 加载图片id
        self.image_path = os.path.join(data_dir, 'AllImages')          # 图片文件夹
        self.label_path = os.path.join(data_dir, 'Annotations')        # 图片对应的标注文件夹

    # 通过txt读取图片的id
    def load_img_ids(self):
        image_set_index_file = os.path.join(self.data_dir, self.phase + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file, 'r') as f:
            lines = f.readlines()
        image_lists = [line.strip() for line in lines]
        return image_lists

    # 读取图片
    def load_image(self, index):
        img_id = self.img_ids[index]
        imgFile = os.path.join(self.image_path, img_id+'.bmp')
        assert os.path.exists(imgFile), 'image {} not existed'.format(imgFile)
        img = cv2.imread(imgFile)
        return img

    # 获取图片id对应的标注xml路径
    def load_annoFolder(self, img_id):
        return os.path.join(self.label_path, img_id+'.xml')

    # 加载标注信息
    def load_annotation(self, index):
        image = self.load_image(index)
        h,w,c = image.shape
        valid_pts = []
        valid_cat = []
        valid_dif = []
        target = ET.parse(self.load_annoFolder(self.img_ids[index])).getroot()
        for obj in target.iter('HRSC_Object'):
            difficult = int(obj.find('difficult').text)
            box_xmin = int(obj.find('box_xmin').text)  # bbox（水平）
            box_ymin = int(obj.find('box_ymin').text)
            box_xmax = int(obj.find('box_xmax').text)
            box_ymax = int(obj.find('box_ymax').text)
            mbox_cx = float(obj.find('mbox_cx').text)  # rbox（旋转）
            mbox_cy = float(obj.find('mbox_cy').text)
            mbox_w = float(obj.find('mbox_w').text)
            mbox_h = float(obj.find('mbox_h').text)
            mbox_ang = float(obj.find('mbox_ang').text)*180/np.pi
            rect = ((mbox_cx, mbox_cy), (mbox_w, mbox_h), mbox_ang)
            pts_4 = cv2.boxPoints(rect)  # 4 x 2      # 五值变八值
            bl = pts_4[0,:]
            tl = pts_4[1,:]
            tr = pts_4[2,:]
            br = pts_4[3,:]
            valid_pts.append([bl, tl, tr, br])
            valid_cat.append(self.cat_ids['ship'])
            valid_dif.append(difficult)
        annotation = {}
        annotation['pts'] = np.asarray(valid_pts, np.float32)
        annotation['cat'] = np.asarray(valid_cat, np.int32)
        annotation['dif'] = np.asarray(valid_dif, np.int32)
        return annotation