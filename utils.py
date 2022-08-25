import os
import numpy as np
from dataset import SketchDataset
import torch
from torch_geometric.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class CELoss(nn.Module):
    ''' Cross Entropy Loss with label smoothing '''
    def __init__(self, label_smooth=None, class_num=137):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num

    def forward(self, pred, target):
        ''' 
        Args:
            pred: prediction of model output [N, M]
            target: ground truth of sampler [N]
        '''
        eps = 1e-12
        
        if self.label_smooth is not None:
            target = F.one_hot(target, self.class_num)

            target = torch.clamp(target.float(), min=self.label_smooth/(self.class_num-1), max=1.0-self.label_smooth)
            loss = -1*torch.sum(target*pred, 1)
        
        else:
            print('Label Smooth is None!')
            import pdb; pdb.set_trace()

        return loss.mean()


def load_data(opt, datasetType='train', shuffle=False):
    data_set = SketchDataset(
        opt=opt,
        root=os.path.join('/data/zhengyixiao/dataset', opt.dataset),
        class_name=opt.class_name,
        split=datasetType,
    )
    data_loader = DataLoader(
        data_set,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        num_workers=opt.num_workers
    )
    return data_loader


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.5)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, opt.epoch, eta_min=opt.eta_min)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def build_record(recode, opt):
    # recode = {}
    # recode['timestamp'] = opt.timestamp
    # recode['class_name'] = opt.class_name
    # recode['net_name'] = opt.net_name
    recode['dataset'] = opt.dataset
    recode['in_feature'] = opt.in_feature
    net_structure = {}
    net_structure['n_blocks'] = opt.n_blocks
    net_structure['channels'] = opt.channels     
    recode['net_structure'] = net_structure

    train_message = {}
    train_message['epoch'] = opt.epoch
    train_message['batch_size'] = opt.batch_size
    train_message['lr'] = opt.lr
    train_message['lr_policy'] = opt.lr_policy
    train_message['lr_decay_iters'] = opt.lr_decay_iters
    train_message['beta1'] = opt.beta1
    if opt.shuffle:
        train_message['shuffle'] = opt.shuffle
        train_message['seed'] = opt.seed
    train_message['dataset'] = opt.dataset
    train_message['train_dataset'] = opt.train_dataset
    train_message['valid_dataset'] = opt.valid_dataset
    train_message['pretrain'] = opt.pretrain
    train_message['trainlist'] = opt.trainlist
    train_message['train_log_name'] = opt.train_log_name
    recode['train_message'] = train_message
    recode['eval_way'] = opt.eval_way
    recode['metric_way'] = opt.metric_way
    recode['comment'] = opt.comment
    return recode
