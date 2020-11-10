import os
import cv2
import loss
import torch
import func_utils
import numpy as np
import torch.nn as nn


# 用于数据分布式操作
def collater(data):
    out_data_dict = {}
    for name in data[0]:
        out_data_dict[name] = []
    for sample in data:
        for name in sample:
            out_data_dict[name].append(torch.from_numpy(sample[name]))
    for name in out_data_dict:
        out_data_dict[name] = torch.stack(out_data_dict[name], dim=0)
    return out_data_dict


# train类
class TrainModule(object):
    def __init__(self, dataset, num_classes, model, decoder, down_ratio):
        torch.manual_seed(317)
        self.dataset = dataset  # 此处的dataset不是简单的名称，表示类别
        self.dataset_phase = {'dota': ['train'],
                              'hrsc': ['train', 'test']}
        self.num_classes = num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 这样写不好！！！！！！！
        self.model = model  # 推理模型前部分
        self.decoder = decoder  # 推理模型后部分
        self.down_ratio = down_ratio  # 下采用率

    def save_model_yan(self, path, epoch, model):
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save({'epoch': epoch,'model_state_dict': state_dict}, path)

    # 保存训练模型
    def save_model(self, path, epoch, model, optimizer):
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict()}, path)

    # 加载需要恢复的检查点
    def load_model(self, model, optimizer, resume, strict=True):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['model_state_dict']
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = model.state_dict()
        if not strict:
            for k in state_dict:
                if k in model_state_dict:
                    if state_dict[k].shape != model_state_dict[k].shape:
                        print('Skip loading parameter {}, required shape{}, ' \
                              'loaded shape{}.'.format(k, model_state_dict[k].shape, state_dict[k].shape))
                        state_dict[k] = model_state_dict[k]
                else:
                    print('Drop parameter {}.'.format(k))
            for k in model_state_dict:
                if not (k in state_dict):
                    print('No param {}.'.format(k))
                    state_dict[k] = model_state_dict[k]
        model.load_state_dict(state_dict, strict=False)  # 恢复检查点
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 恢复优化器设置
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        epoch = checkpoint['epoch']  # 获取当前检查点的epoch
        #  optimizer, 
        return model, optimizer, epoch

    # 训练
    def train_network(self, args):
        self.optimizer = torch.optim.Adam(self.model.parameters(), args.init_lr)
        start_epoch = 1
        path = 'weights_dota/model_last.pth'    # 恢复
        if '.pth' in args.resume:
             self.model,_,start_epoch = self.load_model(self.model,self.optimizer,path,strict =True)
        # 将学习率去掉 self.optimizer
                
        # 以指数形式衰减学习率（可以调参！！！！！！！！！！！）
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.96, last_epoch=-1)
        save_path = 'weights_' + args.dataset  # 创建检测点保存目录

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        # 多GPU并行操作
        if args.ngpus > 1:
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        criterion = loss.LossAll()  # loss函数的设计
        print('Setting up data...')
        dataset_module = self.dataset[args.dataset]  # DOTA类
        dsets = {x: dataset_module(data_dir=args.data_dir,
                                   phase=x,
                                   input_h=args.input_h,
                                   input_w=args.input_w,
                                   down_ratio=self.down_ratio)
                 for x in self.dataset_phase[args.dataset]}  # DOTA类的初始化操作
        # 调用torch自带函数完成数据迭代器的生成
        dsets_loader = {}
        dsets_loader['train'] = torch.utils.data.DataLoader(dsets['train'],
                                                            batch_size=args.batch_size,
                                                            shuffle=True,
                                                            num_workers=args.num_workers,
                                                            pin_memory=True,
                                                            drop_last=True,
                                                            collate_fn=collater)
        print('Starting training...')
        train_loss = []
        ap_list = []
        # 迭代
        for epoch in range(start_epoch, args.num_epoch + 1):
            print('-' * 10)
            print('Epoch: {}/{} '.format(epoch, args.num_epoch))
            # 单个epoch的loss
            epoch_loss = self.run_epoch(phase='train',
                                        data_loader=dsets_loader['train'],
                                        criterion=criterion)
            train_loss.append(epoch_loss)
            self.scheduler.step()  # 根据epoch调整学习率
            np.savetxt(os.path.join(save_path, 'train_loss.txt'), train_loss, fmt='%.6f')  # 保存训练时loss的变化
            # 检查点保存 or epoch > 40
            if epoch % 10 == 0:
                self.save_model_yan(os.path.join(save_path, 'model_{}.pth'.format(epoch)),
                                epoch,
                                self.model)
            '''
            # 如果有test关键字，则进行验证输出
            if 'test' in self.dataset_phase[args.dataset] and epoch % 5 == 0:
                mAP = self.dec_eval(args, dsets['test'])
                ap_list.append(mAP)
                np.savetxt(os.path.join(save_path, 'ap_list.txt'), ap_list, fmt='%.6f')
           '''
            # 保存每个epoch的模型
            self.save_model(os.path.join(save_path, 'model_last.pth'), epoch, self.model, self.optimizer)

    # 单个epoch操作
    def run_epoch(self, phase, data_loader, criterion):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
        running_loss = 0.
        for data_dict in data_loader:
            for name in data_dict:
                data_dict[name] = data_dict[name].to(device=self.device, non_blocking=True)
            if phase == 'train':
                self.optimizer.zero_grad()
                with torch.enable_grad():
                    pr_decs = self.model(data_dict['input'])  # 输入图片（！！！！！！！！）
                    loss = criterion(pr_decs, data_dict)  # 计算loss
                    loss.backward()
                    self.optimizer.step()
            else:
                with torch.no_grad():
                    pr_decs = self.model(data_dict['input'])
                    loss = criterion(pr_decs, data_dict)
            running_loss += loss.item()
        epoch_loss = running_loss / len(data_loader)
        print('{} loss: {}'.format(phase, epoch_loss))
        return epoch_loss

    # 验证函数
    def dec_eval(self, args, dsets):
        result_path = 'result_' + args.dataset
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        self.model.eval()
        func_utils.write_results(args, self.model, dsets, self.down_ratio,
                                 self.device, self.decoder, result_path)
        ap = dsets.dec_evaluation(result_path)
        return ap