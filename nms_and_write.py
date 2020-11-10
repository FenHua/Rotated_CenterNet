import os
import func_utils
import numpy as np


# 数据类别
category = ['baseball-diamond', 'ground-track-field', 'small-vehicle','large-vehicle',
            'tennis-court', 'basketball-court','roundabout']


# 合并成大图的结果，并进行NMS处理
def writeToRAWIMAGE_andNMS(data_dir, image_name, category):
    raw_image_name = image_name
    nms_pts = {cat: [] for cat in category}       # bbox的结果
    nms_scores = {cat: [] for cat in category}    # nms对应的scores
    with open(os.path.join(data_dir, 'result', raw_image_name + '.txt'), 'r') as f:
        # 读raw_image_name对应的图片
        pres = f.readlines()
        if pres == "":
            return                                # 为空则返回
        for pre in pres:
            # 逆时针（不改变顺序）
            pre = pre.strip('\n').split(" ")      # 以换行切断，以空格分割
            pre[4], pre[8] = pre[8], pre[4]
            pre[5], pre[9] = pre[9], pre[5]
            nms_pts[pre[0]].append(pre[2:])       # 更新当前类别的bbox结果
            nms_scores[pre[0]].append(pre[1])     # 记录当前类别的置信度
    nms_result = {cat: [] for cat in category}    # NMS后的结果
    for cat in category:
        if cat in nms_pts:
            nms_pts_cat = np.asarray(nms_pts[cat], np.float32).reshape(-1, 4, 2)       # 转换为4个点
            nms_score_cat = np.asarray(nms_scores[cat], np.float32)                    # scores
            nms_item = func_utils.non_maximum_suppression(nms_pts_cat, nms_score_cat)  # NMS操作（阈值可改！！！！！！！！！）
            if nms_item.shape[0] != 0:
                nms_result[cat].extend(nms_item)
    if not os.path.exists(os.path.join(data_dir, 'After_nms_result')):
        os.mkdir(os.path.join(data_dir, 'After_nms_result'))
    with open(os.path.join(data_dir, 'After_nms_result', raw_image_name + '.txt'), 'w+') as f:
        for cat in nms_result.keys():
            for predict in nms_result[cat]:
                # 又转成顺时针（？？？？？？？？？？？？？？？）
                predict[2], predict[6] = predict[6], predict[2]
                predict[3], predict[7] = predict[7], predict[3]
                location = [str(pre) for pre in predict[:8]]
                f.write(cat + " " + str(predict[-1]) + " " + " ".join(location) + '\n')
