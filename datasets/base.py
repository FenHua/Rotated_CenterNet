import cv2
import math
import torch
import numpy as np
from . import data_augment
import torch.utils.data as data
from .draw_gaussian import draw_umich_gaussian, gaussian_radius
from .transforms import random_flip, load_affine_matrix, random_crop_info, ex_box_jaccard


class BaseDataset(data.Dataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=None):
        super(BaseDataset, self).__init__()
        self.data_dir = data_dir          # 数据目录
        self.phase = phase                # 数据模式
        self.input_h = input_h            # 网络需要的长
        self.input_w = input_w            # 网络需要的宽
        self.down_ratio = down_ratio      # 缩放尺度
        self.img_ids = None               # 图片id
        self.num_classes = None           # 类别
        self.max_objs = 500               # 最大目标数（注意调整！！！！！！！！！！！！！！！！！）
        self.image_distort = data_augment.PhotometricDistort()

    def load_img_ids(self):
        return None

    def load_image(self, index):
        return None

    def load_annoFolder(self, img_id):
        return None

    def load_annotation(self, index):
        return None

    def dec_evaluation(self, result_path):
        return None

    # 附带数据增广的数据转换
    def data_transform(self, image, annotation):
        # only do random_flip augmentation to original images
        crop_size = None
        crop_center = None
        crop_size, crop_center = random_crop_info(h=image.shape[0], w=image.shape[1])
        image, gt_pts, crop_center = random_flip(image, annotation['pts'], crop_center)
        annotation['pts'] = gt_pts
        if crop_center is None:
            crop_center = np.asarray([float(image.shape[1])/2, float(image.shape[0])/2], dtype=np.float32)
        if crop_size is None:
            crop_size = [max(image.shape[1], image.shape[0]), max(image.shape[1], image.shape[0])]  # init
        M = load_affine_matrix(crop_center=crop_center,
                               crop_size=crop_size,
                               dst_size=(self.input_w, self.input_h),
                               inverse=False,
                               rotation=True)   # 基于中心点的随机裁剪操作（不改变图片大小）
        image = cv2.warpAffine(src=image, M=M, dsize=(self.input_w, self.input_h), flags=cv2.INTER_LINEAR)  # 仿射变换（死锁）
        if annotation['pts'].shape[0]:
            annotation['pts'] = np.concatenate([annotation['pts'], np.ones((annotation['pts'].shape[0], annotation['pts'].shape[1], 1))], axis=2)
            annotation['pts'] = np.matmul(annotation['pts'], np.transpose(M))
            annotation['pts'] = np.asarray(annotation['pts'], np.float32)
        out_annotations = {}
        size_thresh = 3
        out_rects = []
        out_cat = []
        for pt_old, cat in zip(annotation['pts'] , annotation['cat']):
            if (pt_old<0).any() or (pt_old[:,0]>self.input_w-1).any() or (pt_old[:,1]>self.input_h-1).any():
                pt_new = pt_old.copy()
                pt_new[:,0] = np.minimum(np.maximum(pt_new[:,0], 0.), self.input_w - 1)
                pt_new[:,1] = np.minimum(np.maximum(pt_new[:,1], 0.), self.input_h - 1)
                iou = ex_box_jaccard(pt_old.copy(), pt_new.copy())
                if iou > 0.8:
                    # 0.8 以上IoU保留
                    rect = cv2.minAreaRect(pt_new/self.down_ratio)
                    # 长宽大于3个像素，考虑
                    if rect[1][0]>size_thresh and rect[1][1]>size_thresh:
                        out_rects.append([rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]])
                        out_cat.append(cat)
            else:
                # 长宽大于3像素，考虑
                rect = cv2.minAreaRect(pt_old/self.down_ratio)
                if rect[1][0]<size_thresh and rect[1][1]<size_thresh:
                    continue
                out_rects.append([rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]])
                out_cat.append(cat)
        out_annotations['rect'] = np.asarray(out_rects, np.float32)
        out_annotations['cat'] = np.asarray(out_cat, np.uint8)
        return image, out_annotations

    # 返回图片数量
    def __len__(self):
        return len(self.img_ids)

    # 输入网络前的图像预处理
    def processing_test(self, image, input_h, input_w):
        image = cv2.resize(image, (input_w, input_h))
        out_image = image.astype(np.float32) / 255.
        out_image = out_image - 0.5
        out_image = out_image.transpose(2, 0, 1).reshape(1, 3, input_h, input_w)
        out_image = torch.from_numpy(out_image)
        return out_image

    # 计算长宽
    def cal_bbox_wh(self, pts_4):
        x1 = np.min(pts_4[:,0])
        x2 = np.max(pts_4[:,0])
        y1 = np.min(pts_4[:,1])
        y2 = np.max(pts_4[:,1])
        return x2-x1, y2-y1

    # 计算bbox信息
    def cal_bbox_pts(self, pts_4):
        x1 = np.min(pts_4[:,0])
        x2 = np.max(pts_4[:,0])
        y1 = np.min(pts_4[:,1])
        y2 = np.max(pts_4[:,1])
        bl = [x1, y2]
        tl = [x1, y1]
        tr = [x2, y1]
        br = [x2, y2]
        return np.asarray([bl, tl, tr, br], np.float32)

    # 对bbox进行排序
    def reorder_pts(self, tt, rr, bb, ll):
        pts = np.asarray([tt,rr,bb,ll],np.float32)
        l_ind = np.argmin(pts[:,0])
        r_ind = np.argmax(pts[:,0])
        t_ind = np.argmin(pts[:,1])
        b_ind = np.argmax(pts[:,1])
        tt_new = pts[t_ind,:]
        rr_new = pts[r_ind,:]
        bb_new = pts[b_ind,:]
        ll_new = pts[l_ind,:]
        return tt_new,rr_new,bb_new,ll_new

    # 计算真实label对应的信息（用于loss计算）
    def generate_ground_truth(self, image, annotation):
        # 图片的加载
        image = np.asarray(np.clip(image, a_min=0., a_max=255.), np.float32)
        image = self.image_distort(np.asarray(image, np.float32))
        image = np.asarray(np.clip(image, a_min=0., a_max=255.), np.float32)
        image = np.transpose(image / 255. - 0.5, (2, 0, 1))
        image_h = self.input_h // self.down_ratio    # 映射到热力图大小(最基本的下采样)
        image_w = self.input_w // self.down_ratio    # # 映射到热力图大小（最基本的下采样）
        num_objs = min(annotation['rect'].shape[0], self.max_objs)  # 对象的个数（小于500）

        # 最浅层的真实信息
        hm_P2 = np.zeros((self.num_classes, image_h, image_w), dtype=np.float32)   # 热力图（每个类别一个（已经进行了缩放））
        wh_P2 = np.zeros((self.max_objs, 10), dtype=np.float32)                    # 坐标信息
        cls_theta_P2 = np.zeros((self.max_objs, 1), dtype=np.float32)              # 最多的类别信息（只管500个）
        reg_P2 = np.zeros((self.max_objs, 2), dtype=np.float32)                    # 回归信息
        ind_P2 = np.zeros((self.max_objs), dtype=np.int64)                         # idx
        reg_mask_P2 = np.zeros((self.max_objs), dtype=np.uint8)                    # mask
        # 逐个根据大小来分配真实信息到热力分层
        for k in range(num_objs):
            rect = annotation['rect'][k, :]
            cen_x, cen_y, bbox_w, bbox_h, theta = rect
            # 根据面积分层阈值卡（YAN）
            area = bbox_h*bbox_w   # 得到面积 （3000，10000）
            # 已经经行了4倍缩放
            radius = gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)))  # 高斯
            radius = max(0, int(radius))    # 面积小于3000，浅层检测
            ct = np.asarray([cen_x, cen_y], dtype=np.float32)                  # 中心
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(hm_P2[annotation['cat'][k]], ct_int, radius)
            ind_P2[k] = ct_int[1] * image_w + ct_int[0]
            reg_P2[k] = ct - ct_int                                               # 偏差信息
            reg_mask_P2[k] = 1
            # generate wh ground_truth
            pts_4 = cv2.boxPoints(((cen_x, cen_y), (bbox_w, bbox_h), theta))  # 4 x 2
            bl = pts_4[0,:]
            tl = pts_4[1,:]
            tr = pts_4[2,:]
            br = pts_4[3,:]
            tt = (np.asarray(tl,np.float32)+np.asarray(tr,np.float32))/2
            rr = (np.asarray(tr,np.float32)+np.asarray(br,np.float32))/2
            bb = (np.asarray(bl,np.float32)+np.asarray(br,np.float32))/2
            ll = (np.asarray(tl,np.float32)+np.asarray(bl,np.float32))/2
            if theta in [-90.0, -0.0, 0.0]:
                tt,rr,bb,ll = self.reorder_pts(tt,rr,bb,ll)
            # rotational channel
            wh_P2[k, 0:2] = tt - ct
            wh_P2[k, 2:4] = rr - ct
            wh_P2[k, 4:6] = bb - ct
            wh_P2[k, 6:8] = ll - ct
            # horizontal channel
            w_hbbox, h_hbbox = self.cal_bbox_wh(pts_4)
            wh_P2[k, 8:10] = 1. * w_hbbox, 1. * h_hbbox
            jaccard_score = ex_box_jaccard(pts_4.copy(), self.cal_bbox_pts(pts_4).copy())
            if jaccard_score<0.95:
                cls_theta_P2[k, 0] = 1

        ret = {'input': image, 'hm_P2': hm_P2, 'reg_mask_P2': reg_mask_P2, 'ind_P2': ind_P2,
               'wh_P2': wh_P2, 'reg_P2': reg_P2, 'cls_theta_P2':cls_theta_P2}
        return ret

    def __getitem__(self, index):
        image = self.load_image(index)       # 加载图片
        image_h, image_w, c = image.shape    # 图片大小信息
        if self.phase == 'test':
            img_id = self.img_ids[index]     # 获取图片id
            image = self.processing_test(image, self.input_h, self.input_w)  # 预处理图片
            return {'image': image,
                    'img_id': img_id,
                    'image_w': image_w,
                    'image_h': image_h}
        elif self.phase == 'train':
            annotation = self.load_annotation(index)   # 加载标注信息
            image, annotation = self.data_transform(image, annotation)  # 数据增广转换
            data_dict = self.generate_ground_truth(image, annotation)   # 数据字典
            return data_dict