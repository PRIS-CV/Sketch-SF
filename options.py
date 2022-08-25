import os
import time
import torch
import json
import argparse
import numpy as np
import random

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
    
    def initialize(self):
        # data params
        self.parser.add_argument('--class-name', type=str, default='256p_centered_creature_none_json_64_NoInitial', 
                                 help='the name of the class to train or test')
        self.parser.add_argument('--points-num', type=int, default=256, 
                                 help='the number of the points in one sketch')
        self.parser.add_argument('--dataset', type=str, default='Seq_CreativeSketch_NoValid', 
                                 help='the name of dataset')
        self.parser.add_argument('--label-smooth', type=float, default=0)
        # networks
        self.parser.add_argument('--net-name', type=str, default='Sketch-Segformer', 
                                 help='the name of the net to use')
        self.parser.add_argument('--in-feature', type=int, default=2, 
                                 help='the number of the feature input to net')
        self.parser.add_argument('--out-segment', type=int, default=17,
                                 help='the number of the labels of the segment')
        self.parser.add_argument('--batch-size', type=int, default=64,
                                 help='intout batch size')
        # for net work
        self.parser.add_argument('--n-blocks', type=int, default=4,
                                 help='')
        self.parser.add_argument('--channels', type=int, default=32,
                                 help='channel in backbone')
        # general params
        self.parser.add_argument('--seed', type=int,
                                 help='if specified, uses seed')
        self.parser.add_argument('--gpu-ids', type=str, default='3', 
                                 help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints-dir', type=str, default='checkpoints', 
                                 help='models are saved here')
        self.parser.add_argument('--pretrain', type=str, default='-', 
                                 help='which pretrain model to load')
        self.parser.add_argument('--which-epoch', type=str, default='latest', 
                                 help='which epoch to load? set to latest to use latest cached model')
        # eval params
        self.parser.add_argument('--eval-way', type=str, default='align', 
                                 help='align or unalign')
        self.parser.add_argument('--metric-way', type=str, default='wlen', 
                                 help='wlen or wolen')

        # other
        self.parser.add_argument('--num-workers', type=int, default=0,
                                 help='')

    def parse(self, params=None):
        if not self.initialized:
            self.initialize()
        self.opt, _ = self.parser.parse_known_args()
        self.opt.is_train = self.is_train

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])
        
        if self.opt.seed is not None:
            # import numpy as np
            # import random
            # torch.manual_seed(self.opt.seed)
            # np.random.seed(self.opt.seed)
            # random.seed(self.opt.seed)
            set_seed(self.opt.seed)
        
        # change opt from params
        if params:
            for key in params.keys():
                setattr(self.opt, key, params[key])
        
        args = vars(self.opt)
        if self.is_train:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

            # self.opt.timestamp = time.strftime("%b%d_%H_%M")
            self.opt.timestamp = time.strftime("%b%d_%H_%M_%S")
            # save to the disk
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.dataset, self.opt.class_name, self.opt.timestamp)
            mkdir(expr_dir)

            # option record
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')

        return self.opt

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--epoch', type=int, default=100,
                                 help='epoch')
        self.parser.add_argument('--lr', type=float, default=0.01, 
                                 help='initial learning rate for adam')
        self.parser.add_argument('--beta1', type=float, default=0.9, 
                                 help='momentum term of adam')
        self.parser.add_argument('--eta-min', type=float, default=0)
        self.parser.add_argument('--weight-decay', type=float, default=5e-4)
        self.parser.add_argument('--lr-policy', type=str, default='cos', 
                                 help='learning rate policy: lambda|step|plateau|cos')
        self.parser.add_argument('--lr-decay-iters', type=int, default=50, 
                                 help='multiply by a gamma every lr_decay_iters iterations')
        self.parser.add_argument("--shuffle", action='store_true', default=True,
                                 help='if shuffle dataset while training')
        self.is_train = True
        self.parser.add_argument("--trainlist", type=str, default='-',
                                 help='the json file name of list to train')
        # self.parser.add_argument("--random-iter", type=int, default=1,
        #                          help='')
        self.parser.add_argument("--train-log-name", type=str, default='-',
                                 help='')                  

        self.parser.add_argument("--train-dataset", type=str, default='train',
                                 help='which dataset to train, train or train2 or trainxxx')
        self.parser.add_argument("--valid-dataset", type=str, default='test',
                                 help='which dataset to valid during training')
        self.parser.add_argument("--comment", type=str, default='-',
                                 help='some comments')

        self.parser.add_argument('--plot-freq', type=int, default=1, 
                                 help='frequency of ploting training loss')
        self.parser.add_argument('--print-freq', type=int, default=10, 
                                 help='frequency of showing training loss on console')
        self.parser.add_argument('--save-epoch-freq', type=int, default=20, 
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--run-test-freq', type=int, default=1, 
                                 help='frequency of running test in training script')

        self.parser.add_argument('--no-vis', action='store_true', 
                                 help='will not use tensorboard')
        self.parser.add_argument('--plot-weights', action='store_true', 
                                 help='plots network weights, etc.')


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--timestamp', type=str, default='-', 
                help='the timestep of the model')
        self.parser.add_argument('--print-freq', type=int, default=2, 
                help='frequency of showing training results on console')
        self.is_train = False


