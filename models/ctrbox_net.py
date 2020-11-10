import torch
import numpy as np
from . import resnet
import torch.nn as nn
from .model_parts import CombinationModule


class CTRBOX(nn.Module):
    def __init__(self, class_number, down_ratio, final_kernel, head_conv):
        super(CTRBOX, self).__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        self.base_network = resnet.resnet101(pretrained=True)
        self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
        self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
        self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True) # 删掉
        self.hm_P2 = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, class_number, kernel_size=final_kernel, stride=1,
                                             padding=final_kernel // 2, bias=True))  # 16！！！！
        self.wh_P2 = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, 10, kernel_size=3, padding=1, bias=True))
        self.reg_P2 = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(head_conv, 2, kernel_size=final_kernel, stride=1,
                                              padding=final_kernel // 2, bias=True))
        self.cls_theta_P2 = nn.Sequential(
            nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 1, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
        # 从初始化bias
        self.fill_fc_weights(self.wh_P2)
        self.fill_fc_weights(self.reg_P2)
        self.fill_fc_weights(self.cls_theta_P2)
        self.hm_P2[-1].bias.data.fill_(-2.19)

    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        c4_combine = self.dec_c4(x[-1], x[-2])  # 删掉
        c3_combine = self.dec_c3(c4_combine, x[-3]) # 将c4_combine 改成x[-2]
        c2_combine = self.dec_c2(c3_combine, x[-4])
        dec_dict = {}
        # 最浅层的检测结果
        dec_dict['hm_P2'] = torch.sigmoid(self.hm_P2(c2_combine))
        dec_dict['wh_P2'] = self.wh_P2(c2_combine)
        dec_dict['reg_P2'] = self.reg_P2(c2_combine)
        dec_dict['cls_theta_P2'] = torch.sigmoid(self.cls_theta_P2(c2_combine))
        return dec_dict