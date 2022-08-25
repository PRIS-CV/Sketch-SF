import os.path as osp
import copy
import json
import math
import random
import torch
import numpy as np
from torch_geometric.data import Data

class SketchData(Data):
    def __init__(self, stroke_idx=None, x=None, edge_index=None, edge_attr=None, y=None,
                 pos=None, norm=None, face=None, **kwargs):
        super(SketchData, self).__init__(x, edge_index, edge_attr, y, pos, norm, face, **kwargs)
        self.stroke_idx = stroke_idx
        self.stroke_num = max(stroke_idx) + 1

    def __inc__(self, key, value):
        if 'index' in key or 'face' in key:
            return self.num_nodes
        elif 'stroke' in key:
            return self.stroke_num
        else:
            return 0

class SketchDataset(torch.utils.data.Dataset):
    def __init__(self, opt, root, class_name, split='train'):
        self.class_name = class_name
        self.split = split
        self.pt_dir = osp.join(root, '{}_{}.pt'.format(self.class_name, self.split))
        self.json_dir = osp.join(root, '{}_{}.ndjson'.format(self.class_name, self.split))
        self.out_segment = opt.out_segment

        if osp.exists(self.pt_dir):
            self.processed_data = torch.load(self.pt_dir)
        else:
            self.processed_data = self._process()

    def __getitem__(self, index):
        return self.processed_data[index]
    
    def __len__(self):
        return len(self.processed_data)
    
    def _process(self):
        raw_data = []
        with open(self.json_dir, 'r') as f:
            for line in f:
                raw_data.append(json.loads(line)["drawing"])
        processed_data = []

        for idx, sketch in enumerate(raw_data):
            sketchArray = [np.array(s) for s in sketch]
            stroke_idx = np.concatenate([np.zeros(len(s[0])) + i for i, s in enumerate(sketchArray)])
            point = np.concatenate([s.transpose()[:,:2] for s in sketchArray])
            # normalize the data (N x 2)
            point = point.astype(np.float)
            max_point = np.max(point, axis=0)
            min_point = np.min(point, axis=0)
            org_point = point
            point = (point - min_point) / (max_point - min_point)
            # point /= 255
            
            # label: c (N,)
            label = np.concatenate([s[2] for s in sketchArray], axis=0) # (N, )

            # edge_index
            edge_index = []
            s = 0
            for stroke in sketchArray:
                # edge_index.append([s,s])
                for i in range(len(stroke[0])-1):
                    edge_index.append([s+i, s+i+1])
                    edge_index.append([s+i+1, s+i])
                # edge_index.append([s,s+len(stroke[0])-1])
                s += len(stroke[0])
            edge_index = np.array(edge_index).transpose()

            # pool_edge_index
            pool_edge_index = []
            s = 0
            for stroke in sketchArray:
                for i in range(len(stroke[0])):
                    pool_edge_index.append([s+i, s+i]) # self loop
                    for j in range(i+1, len(stroke[0])):
                        pool_edge_index.append([s+i, s+j])
                        pool_edge_index.append([s+j, s+i])
                s += len(stroke[0])
            pool_edge_index = np.array(pool_edge_index).transpose()

            sketch_data = SketchData(x=torch.FloatTensor(point),
                                    org_x=torch.FloatTensor(org_point),
                                    edge_index=torch.LongTensor(edge_index),
                                    stroke_idx=torch.LongTensor(stroke_idx),
                                    y=torch.LongTensor(label),
                                    pool_edge_index=torch.LongTensor(pool_edge_index),
                                    )
            processed_data.append(sketch_data)
        torch.save(processed_data, self.pt_dir)
        return processed_data


