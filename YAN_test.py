import os
import cv2
import time
import torch
import func_utils
import numpy as np


# 测试类
class TestModule(object):
    def __init__(self, dataset, num_classes, model, decoder):
        torch.manual_seed(317)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset           # 数据类
        self.num_classes = num_classes   # 类别数
        self.model = model               # 推理模型前部分
        self.decoder = decoder           # 推理模型后部分

    # 加载检查点
    def load_model(self, model, resume):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['model_state_dict']
        model.load_state_dict(state_dict_, strict=True)
        return model

    # 输入网络前的图像预处理
    def processing_test(self, image):
        input_h = 608
        input_w = 608
        image = cv2.resize(image, (input_w, input_h))
        out_image = image.astype(np.float32) / 255.
        out_image = out_image - 0.5
        out_image = out_image.transpose(2, 0, 1).reshape(1, 3, input_h, input_w)
        out_image = torch.from_numpy(out_image)
        return out_image

    # test函数
    def test(self, args, down_ratio):
        save_path = 'weights_'+args.dataset
        self.model = self.load_model(self.model, os.path.join(save_path, args.resume))  # 根据指定位置恢复模型
        self.model = self.model.to(self.device)
        self.model.eval()
        total_time = []
        images = os.listdir('datasets/test/images')    # 读取images文件夹下面的图片，通过滑框形式检测大图
        for i in range(0,len(images)):
            print('processing {}/{} image ...'.format(i, len(images)))
            current_img = cv2.imread(images[i])
            w, h, _ = current_img.shape                # 获取当前图片的大小
            # 获取最佳的重叠率
            if (w<500):
                sx = 0
            else:
                for t in range(0,500):
                    if (w-t)%500:
                        sx =t
            if (h<500):
                sy = 0
            else:
                for t in range(0,500):
                    if (w-t)%500:
                        sy =t


            begin_time = time.time()
            with torch.no_grad():
                pr_decs = self.model(image)
            torch.cuda.synchronize(self.device)
            decoded_pts = []
            decoded_scores = []
            predictions = self.decoder.ctdet_decode(pr_decs)
            pts0, scores0 = func_utils.decode_prediction(predictions, dsets, args, img_id, down_ratio)
            decoded_pts.append(pts0)          # 检测结果的bbox信息
            decoded_scores.append(scores0)    # 检测结果的scores信息
            # 根据类别进行NMS操作
            results = {cat:[] for cat in dsets.category}
            for cat in dsets.category:
                if cat == 'background':
                    continue
                pts_cat = []
                scores_cat = []
                for pts0, scores0 in zip(decoded_pts, decoded_scores):
                    pts_cat.extend(pts0[cat])
                    scores_cat.extend(scores0[cat])
                pts_cat = np.asarray(pts_cat, np.float32)
                scores_cat = np.asarray(scores_cat, np.float32)
                if pts_cat.shape[0]:
                    nms_results = func_utils.non_maximum_suppression(pts_cat, scores_cat)
                    results[cat].extend(nms_results)
            end_time = time.time()
            total_time.append(end_time-begin_time)
            ori_image = dsets.load_image(cnt)
            height, width, _ = ori_image.shape
            # 根据检测结果进行可视化操作
            for cat in dsets.category:
                if cat == 'background':
                    continue
                result = results[cat]
                for pred in result:
                    score = pred[-1]
                    tl = np.asarray([pred[0], pred[1]], np.float32)
                    tr = np.asarray([pred[2], pred[3]], np.float32)
                    br = np.asarray([pred[4], pred[5]], np.float32)
                    bl = np.asarray([pred[6], pred[7]], np.float32)
                    tt = (np.asarray(tl, np.float32) + np.asarray(tr, np.float32)) / 2
                    rr = (np.asarray(tr, np.float32) + np.asarray(br, np.float32)) / 2
                    bb = (np.asarray(bl, np.float32) + np.asarray(br, np.float32)) / 2
                    ll = (np.asarray(tl, np.float32) + np.asarray(bl, np.float32)) / 2
                    box = np.asarray([tl, tr, br, bl], np.float32)
                    cen_pts = np.mean(box, axis=0)
                    cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(tt[0]), int(tt[1])), (0,0,255),1,1)
                    cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(rr[0]), int(rr[1])), (255,0,255),1,1)
                    cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(bb[0]), int(bb[1])), (0,255,0),1,1)
                    cv2.line(ori_image, (int(cen_pts[0]), int(cen_pts[1])), (int(ll[0]), int(ll[1])), (255,0,0),1,1)
                    ori_image = cv2.drawContours(ori_image, [np.int0(box)], -1, (255,0,255),1,1)
                    cv2.putText(ori_image, '{:.2f} {}'.format(score, cat), (box[1][0], box[1][1]),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255), 1,1)
            cv2.imshow('pr_image', ori_image)
            k = cv2.waitKey(0) & 0xFF
            if k == ord('q'):
                cv2.destroyAllWindows()
                exit()
        total_time = total_time[1:]
        print('avg time is {}'.format(np.mean(total_time)))
        print('FPS is {}'.format(1./np.mean(total_time)))