import torch
import torch.nn.functional as F


#检测结果的解码器
class DecDecoder(object):
    def __init__(self, K, conf_thresh, num_classes):
        self.K = K                         # 保留K个目标
        self.conf_thresh = conf_thresh     # 置信度阈值
        self.num_classes = num_classes     # 类别数

    def _topk(self, scores):
        batch, cat, height, width = scores.size()
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), self.K)   # 根据score选最大的K个候选目标
        # 坐标转换
        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds // width).int().float()
        topk_xs = (topk_inds % width).int().float()
        # 针对batch的坐标转换
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), self.K)
        topk_clses = (topk_ind // self.K).int()     # top k对应的类别
        topk_inds = self._gather_feat( topk_inds.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, self.K)
        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    # 热力图
    def _nms(self, heat, kernel=3):
        hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=(kernel - 1) // 2)
        keep = (hmax == heat).float()
        return heat * keep

    # 收集函数
    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    # 数据转换后的收集函数
    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    # 对网络输出结果的解码操作
    def ctdet_decode(self, pr_decs):
        # 分层解析（YAN）
        '''浅层解析P2'''
        heat_P2 = pr_decs['hm_P2']   # 热力图
        wh_P2 = pr_decs['wh_P2']     # 尺度结果
        reg_P2 = pr_decs['reg_P2']   # 偏差回归结果
        cls_theta_P2 = pr_decs['cls_theta_P2']    # 置信度
        batch, c, height, width = heat_P2.size()  # heatmap大小
        heat_P2 = self._nms(heat_P2)              # nms
        scores, inds, clses, ys, xs = self._topk(heat_P2)       # 选出top k个中心点
        reg_P2 = self._tranpose_and_gather_feat(reg_P2, inds)   # 回归信息
        reg_P2 = reg_P2.view(batch, self.K, 2)
        xs = xs.view(batch, self.K, 1) + reg_P2[:, :, 0:1]      # 加上偏差信息
        ys = ys.view(batch, self.K, 1) + reg_P2[:, :, 1:2]
        clses = clses.view(batch, self.K, 1).float()
        scores = scores.view(batch, self.K, 1)
        wh_P2 = self._tranpose_and_gather_feat(wh_P2, inds)
        wh_P2 = wh_P2.view(batch, self.K, 10)
        # add
        cls_theta_P2 = self._tranpose_and_gather_feat(cls_theta_P2, inds)
        cls_theta_P2 = cls_theta_P2.view(batch, self.K, 1)
        mask = (cls_theta_P2>0.8).float().view(batch, self.K, 1)
        tt_x = (xs+wh_P2[..., 0:1])*mask + (xs)*(1.-mask)
        tt_y = (ys+wh_P2[..., 1:2])*mask + (ys-wh_P2[..., 9:10]/2)*(1.-mask)
        rr_x = (xs+wh_P2[..., 2:3])*mask + (xs+wh_P2[..., 8:9]/2)*(1.-mask)
        rr_y = (ys+wh_P2[..., 3:4])*mask + (ys)*(1.-mask)
        bb_x = (xs+wh_P2[..., 4:5])*mask + (xs)*(1.-mask)
        bb_y = (ys+wh_P2[..., 5:6])*mask + (ys+wh_P2[..., 9:10]/2)*(1.-mask)
        ll_x = (xs+wh_P2[..., 6:7])*mask + (xs-wh_P2[..., 8:9]/2)*(1.-mask)
        ll_y = (ys+wh_P2[..., 7:8])*mask + (ys)*(1.-mask)
        # xs: cen_x   ys: cen_y
        detections_P2 = torch.cat([xs, ys, tt_x, tt_y, rr_x, rr_y, bb_x, bb_y, ll_x, ll_y, scores, clses], dim=2)
        index_P2 = (scores>self.conf_thresh).squeeze(0).squeeze(1)   # 根据置信度过滤
        detections = detections_P2[:,index_P2,:]
        # 返回检测结果
        return detections.data.cpu().numpy()