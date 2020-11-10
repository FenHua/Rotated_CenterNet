import os
import eval
import test
import train
import decoder
import argparse
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from models import ctrbox_net             # 加载CenterNet
from datasets.dataset_dota import DOTA    # 加载DOTA数据接口
from datasets.dataset_hrsc import HRSC    # 加载HRSC数据接口


def parse_args():
    parser = argparse.ArgumentParser(description='BBDet')
    parser.add_argument('--num_epoch', type=int, default=1, help='Number of epochs')      # 迭代次数
    parser.add_argument('--batch_size', type=int, default=1, help='Number of epochs')     # batch_size 大小
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--init_lr', type=float, default=3e-5, help='Init learning rate')     # 
    parser.add_argument('--input_h', type=int, default=800, help='Resized image height')
    parser.add_argument('--input_w', type=int, default=800, help='Resized image width')
    parser.add_argument('--K', type=int, default=200, help='maximum of objects')         # 每张切片保留K个目标进行判断  500
    parser.add_argument('--conf_thresh', type=float, default=0.18, help='threshold')     # 置信度
    parser.add_argument('--ngpus', type=int, default=2, help='number of gpus')           # GPU个数
    parser.add_argument('--resume', type=str, default='model_last.pth', help='weights to be resumed')   # 检查点位置
    parser.add_argument('--dataset', type=str, default='hrsc', help='weights to be resumed')       # 数据集类别（去掉） dota
    parser.add_argument('--data_dir', type=str, default='../datasets/dota', help='data directory')   # Dataset path
    parser.add_argument('--phase', type=str, default='test', help='data directory')      # 执行模式
    parser.add_argument('--wh_channels', type=int, default=8, help='data directory')     #
    parser.add_argument('--out_dir', type=str, default='10250930', help='data output directory')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    dataset = {'dota': DOTA, 'hrsc': HRSC}
    num_classes = {'dota': 5, 'hrsc': 1}    # 五类
    heads = {'hm': num_classes[args.dataset], 'wh': 10, 'reg': 2, 'cls_theta': 1 }   # 参数设置（对应不同网络位置处的通道数）
    down_ratio = 4       # 特征图相对原图下采样幅度
    class_number = heads['hm']      # 读取需要分类的类别个数
    # phi = 2           # 确定网络的大小
    #input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    #args.input_h = input_sizes[phi]  # 根据phi值更新输入图片的大小
    #args.input_w = input_sizes[phi]
    model = ctrbox_net.CTRBOX(class_number, down_ratio=down_ratio, final_kernel=1, head_conv=64)   # 构建CenterNet  32
    decoder = decoder.DecDecoder(K=args.K, conf_thresh=args.conf_thresh, num_classes=num_classes[args.dataset])     # 对网络输出结果的解码
    if args.phase == 'train':
        # 训练阶段
        ctrbox_obj = train.TrainModule(dataset=dataset, num_classes=num_classes, model=model, decoder=decoder, down_ratio=down_ratio)
        ctrbox_obj.train_network(args)
    elif args.phase == 'test':
        # 测试
        ctrbox_obj = test.TestModule(dataset=dataset, num_classes=num_classes, model=model, decoder=decoder)
        ctrbox_obj.test(args, down_ratio=down_ratio)
    else:
        # 验证模式
        ctrbox_obj = eval.EvalModule(dataset=dataset, num_classes=num_classes, model=model, decoder=decoder)
        ctrbox_obj.evaluation(args, down_ratio=down_ratio)