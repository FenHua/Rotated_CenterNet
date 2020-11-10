import os
import cv2
import time
import glob
import torch
import func_utils
import numpy as np
import nms_and_write


# 获取切片相对于大图的位置
def getLT(data_dir):
    LT_dict = {}
    with open(os.path.join(data_dir, 'relate_test.txt'), 'r') as f:
        for line in f.readlines():
            line = line.split(',')
            LT_dict[line[0]] = (line[1], line[2])    # 只调用了左上角的坐标
    if not os.path.exists(os.path.join(data_dir, 'result')):
        os.mkdir(os.path.join(data_dir, 'result'))
    return LT_dict


# 比赛结果逆时针 [float类型？]
# 输出的txt格式为：类别名称 置信度 目标四点坐标（左上-左下-右下-右上顺序，以空格为间隔）
# 定义类别数
def writeToRAWIMAGE(result, cat, data_dir, image_name, LT_dict):
    raw_image_name, _ = image_name.split('_')   # 获取大图名称
    crop_x1, crop_y1 = LT_dict[image_name]      # 切片相对大图的位置
    with open(os.path.join(data_dir, 'result', raw_image_name + '.txt'), 'a+') as f:
        for pre in result:
            pre[[0, 2, 4, 6]] += float(crop_x1)
            pre[[1, 3, 5, 7]] += float(crop_y1)
            reverse_pre = [str(int(pre[i])) for i in [0, 1, 6, 7, 4, 5, 2, 3]]   # 逆时针（不对）
            f.write(cat + " " + str(pre[-1]) + " " + " ".join(reverse_pre) + '\n')


class TestModule(object):
    def __init__(self, dataset, num_classes, model, decoder):
        torch.manual_seed(317)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.num_classes = num_classes
        self.model = model
        self.decoder = decoder

    # 加载检查点
    def load_model(self, model, resume):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['model_state_dict']
        model.load_state_dict(state_dict_, strict=True)
        return model


    def test(self, args, down_ratio):
        save_path = 'weights_' + args.dataset     # 检查点位置
        self.model = self.load_model(self.model, os.path.join(save_path, args.resume))
        self.model = self.model.to(self.device)
        self.model.eval()
        dataset_module = self.dataset[args.dataset]
        dsets = dataset_module(data_dir=args.data_dir,
                               phase='test',
                               input_h=args.input_h,
                               input_w=args.input_w,
                               down_ratio=down_ratio)
        data_loader = torch.utils.data.DataLoader(dsets,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1,
                                                  pin_memory=True)
        total_time = []
        LT_dict = getLT(args.data_dir)       # 切片与大图之间的位置映射
        for cnt, data_dict in enumerate(data_loader):
            image = data_dict['image'][0].to(self.device)
            img_id = data_dict['img_id'][0]
            print('processing {}/{} image ...'.format(cnt, len(data_loader)))
            begin_time = time.time()
            with torch.no_grad():
                pr_decs = self.model(image)
            torch.cuda.synchronize(self.device)
            decoded_pts = []
            decoded_scores = []
            predictions = self.decoder.ctdet_decode(pr_decs)
            pts0, scores0 = func_utils.decode_prediction(predictions, dsets, args, img_id, down_ratio)
            decoded_pts.append(pts0)
            decoded_scores.append(scores0)
            # 切片中的nms
            results = {cat: [] for cat in dsets.category}
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
            total_time.append(end_time - begin_time)
            # 切片bbox映射到大图
            for cat in dsets.category:
                if cat == 'background':
                    continue
                result = results[cat]
                if result != []:
                    result = np.array(result).astype(np.float32)
                    writeToRAWIMAGE(result, cat, args.data_dir, img_id, LT_dict)
                else:
                    # 没有目标写入空的txt
                    raw_image_name, _ = img_id.split('_')
                    with open(os.path.join(args.data_dir, 'result', raw_image_name + '.txt'), 'a+') as f:
                        pass
        # 对大图上的结果进行NMS
        txt_list = glob.glob(os.path.join(args.data_dir, 'result', '*.txt'))    # 查找符合条件的文档
        for txt in txt_list:
            nms_image_name, _ = os.path.basename(txt).split(".")
            nms_and_write.writeToRAWIMAGE_andNMS(args.data_dir, nms_image_name, dsets.category)  # NMS
        total_time = total_time[1:]
        print('avg time is {}'.format(np.mean(total_time)))
        print('FPS is {}'.format(1. / np.mean(total_time)))