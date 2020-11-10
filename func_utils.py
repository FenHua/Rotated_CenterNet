import os
import torch
import numpy as np
# 调用API进行NMS 处理
from datasets.DOTA_devkit.ResultMerge_multi_process import py_cpu_nms_poly_fast, py_cpu_nms_poly


# 解析预测结果
def decode_prediction(predictions, dsets, args, img_id, down_ratio):
    predictions = predictions[0, :, :]
    ori_image = dsets.load_image(dsets.img_ids.index(img_id))  # 加载数据集
    h, w, c = ori_image.shape                                  # 图片大小
    pts0 = {cat: [] for cat in dsets.category}                 # 类别元组
    scores0 = {cat: [] for cat in dsets.category}              # 类别元组
    for pred in predictions:
        cen_pt = np.asarray([pred[0], pred[1]], np.float32)    # 中心点
        tt = np.asarray([pred[2], pred[3]], np.float32)
        rr = np.asarray([pred[4], pred[5]], np.float32)
        bb = np.asarray([pred[6], pred[7]], np.float32)
        ll = np.asarray([pred[8], pred[9]], np.float32)
        tl = tt + ll - cen_pt
        bl = bb + ll - cen_pt
        tr = tt + rr - cen_pt
        br = bb + rr - cen_pt
        score = pred[10]          # 置信度
        clse = pred[11]           # 类别
        pts = np.asarray([tr, br, bl, tl], np.float32)
        pts[:, 0] = pts[:, 0] * down_ratio / args.input_w * w    # 按照缩放比列进行缩放
        pts[:, 1] = pts[:, 1] * down_ratio / args.input_h * h
        pts0[dsets.category[int(clse)]].append(pts)
        scores0[dsets.category[int(clse)]].append(score)
    return pts0, scores0


# 调用DOTA工具包进行NMS处理
def non_maximum_suppression(pts, scores):
    nms_item = np.concatenate([pts[:, 0:1, 0], pts[:, 0:1, 1], pts[:, 1:2, 0], pts[:, 1:2, 1],
                               pts[:, 2:3, 0], pts[:, 2:3, 1], pts[:, 3:4, 0], pts[:, 3:4, 1], scores[:, np.newaxis]], axis=1)
    nms_item = np.asarray(nms_item, np.float64)
    keep_index = py_cpu_nms_poly_fast(dets=nms_item, thresh=0.3)  # NMS处理（可能需要调参？？？？？？？？？？？？？）
    return nms_item[keep_index]


# 执行整个检测过程
def write_results(args, model, dsets, down_ratio, device, decoder, result_path, print_ps=False):
    results = {cat: {img_id: [] for img_id in dsets.img_ids} for cat in dsets.category}
    # 执行检测过程
    for index in range(len(dsets)):
        data_dict = dsets.__getitem__(index)
        image = data_dict['image'].to(device)
        img_id = data_dict['img_id']
        image_w = data_dict['image_w']
        image_h = data_dict['image_h']
        with torch.no_grad():
            pr_decs = model(image)    # 网络推理
        decoded_pts = []              # 检测候选框信息
        decoded_scores = []           # 置信度
        torch.cuda.synchronize(device)
        predictions = decoder.ctdet_decode(pr_decs)  # CenterNet解析
        pts0, scores0 = decode_prediction(predictions, dsets, args, img_id, down_ratio)    # 预测结果解析
        decoded_pts.append(pts0)
        decoded_scores.append(scores0)
        # 根据类别进行NMS
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
                nms_results = non_maximum_suppression(pts_cat, scores_cat)   # NMS处理后的结果
                results[cat][img_id].extend(nms_results)                     # 不同类别不同图片对应的检测结果
        if print_ps:
            # 是否显示执行进度
            print('testing {}/{} data {}'.format(index+1, len(dsets), img_id))
    # 根据不同的类别分别存储不同的检测结果
    for cat in dsets.category:
        if cat == 'background':
            continue
        with open(os.path.join(result_path, 'Task1_{}.txt'.format(cat)), 'w') as f:
            for img_id in results[cat]:
                for pt in results[cat][img_id]:
                    f.write('{} {:.12f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                        img_id, pt[8], pt[0], pt[1], pt[2], pt[3], pt[4], pt[5], pt[6], pt[7]))
