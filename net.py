import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks.selfatt import *
import torch_geometric.nn as tgnn
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_cluster import knn_graph
# import torchsnooper
import torch_geometric.nn as tgnn
from torch_scatter import scatter_add


def init_net(opt):
    if opt.net_name == 'Sketch-Segformer':
        net = Sketch_Segformer(opt)
    else:
        raise NotImplementedError('net {} is not implemented. Please check.\n'.format(opt.net_name))
    
    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(opt.gpu_ids[0])
        net = net.cuda()
        net = torch.nn.DataParallel(net, device_ids=opt.gpu_ids)
    return net


class Sketch_Segformer(nn.Module):
    def __init__(self, opt):
        super(Sketch_Segformer, self).__init__()
        self.points_num = opt.points_num
        self.channels = opt.channels * 2
        self.in_feature = opt.in_feature
        self.out_segment = opt.out_segment
        self.pos_encoding = nn.Embedding(self.points_num, self.channels)
        self.conv1 = nn.Conv1d(self.in_feature, self.channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(self.channels, self.channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.channels)
        self.bn2 = nn.BatchNorm1d(self.channels)

        self.g_sa1 = GA_Module(self.channels, self.channels//2//4, self.channels//2)
        self.g_down1_trans = nn.Sequential(nn.Conv1d(self.channels, self.channels//2, kernel_size=1, bias=False),
                                nn.BatchNorm1d(self.channels//2),
                                )
        self.s_sa1 = SA_Module(self.channels, self.channels//2//4, self.channels//2)
        self.s_down1_trans = nn.Sequential(nn.Conv1d(self.channels, self.channels//2, kernel_size=1, bias=False),
                                nn.BatchNorm1d(self.channels//2),
                                )
        self.g_sa1_norm = nn.BatchNorm1d(self.channels//2)
        self.s_sa1_norm = nn.BatchNorm1d(self.channels//2)
        self.sa1_trans = nn.Sequential(nn.Conv1d(self.channels, self.channels*4, kernel_size=1, bias=False),
                                nn.BatchNorm1d(self.channels*4),
                                nn.ReLU(inplace=True),
                                nn.Conv1d(self.channels*4, self.channels, kernel_size=1, bias=False),
                                nn.BatchNorm1d(self.channels),
                                )
        self.g_sa2 = GA_Module(self.channels, self.channels//2//4, self.channels//2)
        self.g_down2_trans = nn.Sequential(nn.Conv1d(self.channels, self.channels//2, kernel_size=1, bias=False),
                                nn.BatchNorm1d(self.channels//2),
                                )
        self.s_sa2 = SA_Module(self.channels, self.channels//2//4, self.channels//2)
        self.s_down2_trans = nn.Sequential(nn.Conv1d(self.channels, self.channels//2, kernel_size=1, bias=False),
                                nn.BatchNorm1d(self.channels//2),
                                )
        self.g_sa2_norm = nn.BatchNorm1d(self.channels//2)
        self.s_sa2_norm = nn.BatchNorm1d(self.channels//2)
        self.sa2_trans = nn.Sequential(nn.Conv1d(self.channels, self.channels*4, kernel_size=1, bias=False),
                                nn.BatchNorm1d(self.channels*4),
                                nn.ReLU(inplace=True),
                                nn.Conv1d(self.channels*4, self.channels, kernel_size=1, bias=False),
                                nn.BatchNorm1d(self.channels),
                                )
        self.g_sa3 = GA_Module(self.channels, self.channels//2//4, self.channels//2)
        self.g_down3_trans = nn.Sequential(nn.Conv1d(self.channels, self.channels//2, kernel_size=1, bias=False),
                                nn.BatchNorm1d(self.channels//2),
                                )
        self.s_sa3 = SA_Module(self.channels, self.channels//2//4, self.channels//2)
        self.s_down3_trans = nn.Sequential(nn.Conv1d(self.channels, self.channels//2, kernel_size=1, bias=False),
                                nn.BatchNorm1d(self.channels//2),
                                )
        self.g_sa3_norm = nn.BatchNorm1d(self.channels//2)
        self.s_sa3_norm = nn.BatchNorm1d(self.channels//2)
        self.sa3_trans = nn.Sequential(nn.Conv1d(self.channels, self.channels*4, kernel_size=1, bias=False),
                                nn.BatchNorm1d(self.channels*4),
                                nn.ReLU(inplace=True),
                                nn.Conv1d(self.channels*4, self.channels, kernel_size=1, bias=False),
                                nn.BatchNorm1d(self.channels),
                                )
        self.g_sa4 = GA_Module(self.channels, self.channels//2//4, self.channels//2)
        self.g_down4_trans = nn.Sequential(nn.Conv1d(self.channels, self.channels//2, kernel_size=1, bias=False),
                                nn.BatchNorm1d(self.channels//2),
                                )
        self.s_sa4 = SA_Module(self.channels, self.channels//2//4, self.channels//2)
        self.s_down4_trans = nn.Sequential(nn.Conv1d(self.channels, self.channels//2, kernel_size=1, bias=False),
                                nn.BatchNorm1d(self.channels//2),
                                )
        self.g_sa4_norm = nn.BatchNorm1d(self.channels//2)
        self.s_sa4_norm = nn.BatchNorm1d(self.channels//2)
        self.sa4_trans = nn.Sequential(nn.Conv1d(self.channels, self.channels*4, kernel_size=1, bias=False),
                                nn.BatchNorm1d(self.channels*4),
                                nn.ReLU(inplace=True),
                                nn.Conv1d(self.channels*4, self.channels, kernel_size=1, bias=False),
                                nn.BatchNorm1d(self.channels),
                                )
        self.relu = nn.ReLU(inplace=True)

        self.conv_fuse = nn.Sequential(nn.Conv1d(self.channels*4, self.channels*4*4, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(self.channels*4*4),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Conv1d(self.channels*4*4*3, self.channels*4, kernel_size=1)
        self.bn6 = nn.BatchNorm1d(self.channels*4)
        self.linear2 = nn.Conv1d(self.channels*4, self.channels*4//2, kernel_size=1)
        self.bn7 = nn.BatchNorm1d(self.channels*4//2)
        self.linear3 = nn.Conv1d(self.channels*4//2, opt.out_segment, kernel_size=1)
        # softmax        
        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, edge_index, data):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        edge_index = torch.unique(edge_index, dim=1)
        pool_edge_index, _ = add_self_loops(data['pool_edge_index'], num_nodes=x.size(0))
        pool_edge_index = torch.unique(pool_edge_index, dim=1)

        if len(x.shape) == 2:
            x = x.view(-1, self.points_num, self.in_feature)
        else:
            print("Attention!!!")
            import pdb; pdb.set_trace()
        x = x.permute(0, 2, 1)
        batch_size, _, N = x.size()
        pos = torch.arange(0, self.points_num).repeat(batch_size).view(batch_size, self.points_num).to(device=x.device)
        pos = self.pos_encoding(pos)
        pos = pos.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x))) # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))
        x = x + pos

        g_x1 = self.g_sa1_norm(self.g_sa1(x)) + self.g_down1_trans(x)
        s_x1 = self.s_sa1_norm(self.s_sa1(x.permute(0, 2, 1).contiguous().view(-1, self.channels), pool_edge_index).view(batch_size, self.points_num, -1).permute(0, 2, 1)) + self.s_down1_trans(x)
        x1 = torch.cat((g_x1, s_x1), dim=1)
        x1 = self.relu(self.sa1_trans(x1) + x1)
        torch.cuda.empty_cache()
        g_x2 = self.g_sa2_norm(self.g_sa2(x1)) + self.g_down2_trans(x1)
        s_x2 = self.s_sa2_norm(self.s_sa2(x1.permute(0, 2, 1).contiguous().view(-1, self.channels), pool_edge_index).view(batch_size, self.points_num, -1).permute(0, 2, 1)) + self.s_down2_trans(x1)
        x2 = torch.cat((g_x2, s_x2), dim=1)
        x2 = self.relu(self.sa2_trans(x2) + x2)
        torch.cuda.empty_cache()
        g_x3 = self.g_sa3_norm(self.g_sa3(x2)) + self.g_down3_trans(x2)
        s_x3 = self.s_sa3_norm(self.s_sa3(x2.permute(0, 2, 1).contiguous().view(-1, self.channels), pool_edge_index).view(batch_size, self.points_num, -1).permute(0, 2, 1)) + self.s_down3_trans(x2)
        x3 = torch.cat((g_x3, s_x3), dim=1)
        x3 = self.relu(self.sa3_trans(x3) + x3)
        torch.cuda.empty_cache()
        g_x4 = self.g_sa4_norm(self.g_sa4(x3)) + self.g_down4_trans(x3)
        s_x4 = self.s_sa4_norm(self.s_sa4(x3.permute(0, 2, 1).contiguous().view(-1, self.channels), pool_edge_index).view(batch_size, self.points_num, -1).permute(0, 2, 1)) + self.s_down4_trans(x3)
        x4 = torch.cat((g_x4, s_x4), dim=1)
        x4 = self.relu(self.sa4_trans(x4) + x4)
        torch.cuda.empty_cache()

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_fuse(x)
        x_max = torch.max(x, 2)[0]
        x_avg = torch.mean(x, 2)
        x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, self.points_num)
        x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, self.points_num)
        x = torch.cat((x, x_max_feature, x_avg_feature), 1) # 1024 * 3
        x = F.relu(self.bn6(self.linear1(x)))
        x = F.relu(self.bn7(self.linear2(x)))
        x = self.linear3(x)

        x = x.permute(0, 2, 1)
        x = x.contiguous().view(-1, self.out_segment)
        x = self.LogSoftmax(x)

        return x


