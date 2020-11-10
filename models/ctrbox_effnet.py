import torch
import numpy as np
import torch.nn as nn
from .model_parts import CombinationModule
from .efficientnet import EfficientNet as EffNet


class EfficientNet(nn.Module):
    def __init__(self, phi, load_weights=False):
        super(EfficientNet, self).__init__()
        model = EffNet.from_pretrained(f'efficientnet-b{phi}', load_weights)   # 采用预训练模型
        del model._conv_head       # 删除
        del model._bn1             # 删除
        del model._avg_pooling     # 删除
        del model._dropout         # 删除
        del model._fc              # 删除
        self.model = model

    def forward(self, x):
        x = self.model._conv_stem(x)   # 卷积头操作
        x = self.model._bn0(x)         # BN操作
        x = self.model._swish(x)       # Swish操作
        feature_maps = []              # 存储的特征层
        last_x = None
        for idx, block in enumerate(self.model._blocks):
            # 分别遍历每个block
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)   # 当前block操作
            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(last_x)                     # 特征图缩减处存储特征层
            elif idx == len(self.model._blocks) - 1:
                feature_maps.append(x)                          # 最后的特征层
            last_x = x    # 多余（删）
        del last_x        # 删
        out_feats = [feature_maps[1],feature_maps[2],feature_maps[3],feature_maps[4]]   # 取第2, 3，4，5层对应的特征图
        return out_feats

# CenterNet模型
class CTRBOX(nn.Module):
    def __init__(self, class_number, phi, final_kernel, head_conv):
        super(CTRBOX, self).__init__()
        self.base_network = EfficientNet(phi=phi, load_weights = True)
        self.out_filters = {
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [80, 224, 640],
            }[phi]
        self.dec_c3 = CombinationModule(self.out_filters[1], self.out_filters[0], batch_norm=True)      # 跳连
        self.dec_c4 = CombinationModule(self.out_filters[2], self.out_filters[1], batch_norm=True)
        # 最浅层
        self.hm_P2 = nn.Sequential(nn.Conv2d(self.out_filters[0], head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, class_number, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))  # 16！！！！
        self.wh_P2 = nn.Sequential(nn.Conv2d(self.out_filters[0], head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, 10, kernel_size=3, padding=1, bias=True))
        self.reg_P2 = nn.Sequential(nn.Conv2d(self.out_filters[0], head_conv, kernel_size=3, padding=1, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(head_conv, 2, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
        self.cls_theta_P2 = nn.Sequential(nn.Conv2d(self.out_filters[0], head_conv, kernel_size=3, padding=1, bias=True),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, bias=True),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(head_conv, 1, kernel_size=final_kernel, stride=1, padding=final_kernel // 2,bias=True))
        # 从初始化bias
        self.fill_fc_weights(self.wh_P2)
        self.fill_fc_weights(self.reg_P2)
        self.fill_fc_weights(self.cls_theta_P2)
        self.hm_P2[-1].bias.data.fill_(-2.19)

    # 初始化操作
    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        C4 = self.dec_c4(x[-1],x[-2])
        C3 = self.dec_c3( C4, x[-3])
        # x[0]  1*48*52*52
        # x[1] 1*120*26*26
        # x[2] 1*352*13*13
        dec_dict = {}
        # 最浅层的检测结果
        dec_dict['hm_P2'] = torch.sigmoid(self.hm_P2(C3))
        dec_dict['wh_P2'] = self.wh_P2(C3)
        dec_dict['reg_P2'] = self.reg_P2(C3)
        dec_dict['cls_theta_P2'] = torch.sigmoid(self.cls_theta_P2(C3))
        return dec_dict