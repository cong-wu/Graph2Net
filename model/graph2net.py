import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math


class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        print("Mish activation loaded...")

    def forward(self, x): 
        x = x *( torch.tanh(F.softplus(x)))
        return x

    
def conv_dw(in_planes, out_planes, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=bias)
    
    
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    #nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn_lite(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, groups=1):
        super(unit_tcn_lite, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1), groups=groups, bias=False)
        self.pointwise  = nn.Conv2d(in_channels, out_channels, 1, bias =False)
        self.bn = nn.BatchNorm2d(out_channels)
        #self.relu = nn.ReLU()
        conv_init(self.conv)
        conv_init(self.pointwise)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.pointwise(self.conv(x))
        return self.bn(x)
        
class unit_tcn_lite_new(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(unit_tcn_lite_new, self).__init__()
        
        if in_channels % 2 ==0:
            inter_channel = in_channels // 2
        else:
            raise ValueError("in_channels can not devide by 4")

        self.conv3 = nn.Conv2d(inter_channel, inter_channel, kernel_size=(5, 5), padding=(2, 2),
                              stride=(stride, 1), groups=groups // 2, bias=False)
        self.bn3 = nn.BatchNorm2d(inter_channel)
        self.conv4 = nn.Conv2d(inter_channel, inter_channel, kernel_size=(9, 1), padding=(4, 0),
                              stride=(stride, 1), groups=groups // 2, bias=False)
        self.bn4 = nn.BatchNorm2d(inter_channel)
        
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias =False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = torch.chunk(x, 2, 1)
        ys = []
        
        y = self.relu(self.bn3(self.conv3(x[0])))
        ys.append(y)
        
        y = self.relu(self.bn4(self.conv4(x[1])))
        ys.append(y)
        
        ys = torch.cat(ys, 1)
        
        ys = self.bn(self.conv(ys))
        
        return ys
        
        

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1), bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, num_subset=3):
        super(unit_gcn, self).__init__()
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)

        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_g = nn.Conv2d(in_channels * 3, out_channels, 1, bias=False)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x
       
        self.relu = nn.ReLU(inplace=True)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, in_channels, 1))
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)
        
        self.bn = nn.BatchNorm2d(out_channels)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        X = x.view(N, C * T, V)

        y = None
        for i in range(self.num_subset):

            A1 = A[i]
            temp = self.conv_d[i](torch.matmul(X, A1).view(N, C, T, V))
            if y is None:
                y = temp
            else:
                y = torch.cat((y, temp), 1)
        y = self.conv_g(y)

        y = self.bn(y)

        y += self.down(x)
        return self.relu(y)


class unit_gcn_all(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(unit_gcn_all, self).__init__()
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32))[0])
        nn.init.constant_(self.PA, 1.0/self.PA.size(0))

        self.conv_g = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x
       
        self.relu = nn.ReLU(inplace=True)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
                
        self.bn = nn.BatchNorm2d(out_channels)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.PA

        X = x.view(N, C * T, V)

        y = torch.matmul(X, A).view(N, C, T, V)

        y = self.bn(self.conv_g(y))

        y += self.down(x)
        return self.relu(y)
        

class unit_gcn_2net(nn.Module):
    def __init__(self, in_channels, out_channels, A, num_subset=3):
        super(unit_gcn_2net, self).__init__()
        
        if in_channels % 4 ==0:
            inter_channel = in_channels // 4
        else:
            raise ValueError("in_channels can not devide by 4")
                    
        self.out_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
       
        #first
        self.PA1 = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA1, 1e-6)

        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset
        
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(inter_channel, inter_channel, 1))
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)
        
        self.conv_g = nn.Conv2d(inter_channel * 3, inter_channel, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(inter_channel)
        conv_init(self.conv_g)
        bn_init(self.bn3, 1)
        
        #second
        self.PA2 = nn.Parameter(torch.from_numpy(A.astype(np.float32))[0])
        nn.init.constant_(self.PA2, 1.0/self.PA2.size(0))
        
        self.conv_f = nn.Conv2d(inter_channel, inter_channel, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(inter_channel)
        conv_init(self.conv_f)
        bn_init(self.bn4, 1)
        
        #three
        self.PA3 = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA3, 1e-6)

        self.conv_h = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_h.append(nn.Conv2d(inter_channel, inter_channel, 1))
        for i in range(self.num_subset):
            conv_branch_init(self.conv_h[i], self.num_subset)
        
        self.conv_k = nn.Conv2d(inter_channel * 3, inter_channel, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(inter_channel)
        conv_init(self.conv_k)
        bn_init(self.bn5, 1)
                
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
                
        xs = torch.chunk(x, 4, 1)
        ys = []
        N, C, T, V = xs[0].size()
        
        ys.append(xs[0])
        
        #first
        A = self.A.cuda(x.get_device())
        A = A + self.PA1
        X = xs[1].view(N, C * T, V)
        y = None
        for i in range(self.num_subset):
            A1 = A[i]
            temp = self.conv_d[i](torch.matmul(X, A1).view(N, C, T, V))
            if y is None:
                y = temp
            else:
                y = torch.cat((y, temp), 1)
        y = self.relu(self.bn3(self.conv_g(y)))
        ys.append(y)
        
        #second
        A = self.PA2
        # N, C, T, V = xs[1].size()
        X = (xs[2] + ys[-1]).view(N, C * T, V)
        y = torch.matmul(X, A).view(N, C, T, V)
        y = self.relu(self.bn4(self.conv_f(y)))
        ys.append(y)
        
        
        #three
        A = self.A.cuda(x.get_device())
        A = A + self.PA3
        X = (xs[3] + ys[-1]).view(N, C * T, V)
        y = None
        for i in range(self.num_subset):
            A1 = A[i]
            temp = self.conv_h[i](torch.matmul(X, A1).view(N, C, T, V))
            if y is None:
                y = temp
            else:
                y = torch.cat((y, temp), 1)
        y = self.relu(self.bn5(self.conv_k(y)))
        ys.append(y)
        
        out = torch.cat(ys, 1)
        
        out = self.bn2(self.out_conv(out))
        out += self.down(x)
        return self.relu(out)
        
        
        
class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, groups=1):
        super(TCN_GCN_unit, self).__init__()
        if not residual:
            self.flag = 1
            self.gcn1 = unit_gcn(in_channels, out_channels, A)
            self.gcn2 = unit_gcn_all(in_channels, out_channels, A)
            self.tcn1 = unit_tcn_lite(out_channels, out_channels, stride=stride, groups=groups)
        else:
            self.flag = 0
            self.gcn1 = unit_gcn_2net(in_channels, out_channels, A)
            self.tcn1 = unit_tcn_lite_new(out_channels, out_channels, stride=stride, groups=groups)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn_lite(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups)

    def forward(self, x):
        if self.flag == 1:
            x = self.tcn1(self.gcn1(x) + self.gcn2(x)) + self.residual(x)
        else:
            x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(3, 64, A, residual=False, groups=1)
        self.l2 = TCN_GCN_unit(64, 64, A, groups=64)
        self.l3 = TCN_GCN_unit(64, 64, A, groups=64)
        self.l4 = TCN_GCN_unit(64, 64, A, groups=64)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, groups=64)
        self.l6 = TCN_GCN_unit(128, 128, A, groups=128)
        self.l7 = TCN_GCN_unit(128, 128, A, groups=128)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2, groups=128)
        self.l9 = TCN_GCN_unit(256, 256, A, groups=256)
        self.l10 = TCN_GCN_unit(256, 256, A, groups=256)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x1 = self.l1(x)
        x1 = self.l2(x1)
        x1 = self.l3(x1)
        x1 = self.l4(x1)
        
        x2 = self.l5(x1)
        x2 = self.l6(x2)
        x2 = self.l7(x2)
        
        x3 = self.l8(x2)
        x3 = self.l9(x3)
        x3 = self.l10(x3)

        # N*M,C,T,V
        c_new = x3.size(1)
        x = x3.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)
