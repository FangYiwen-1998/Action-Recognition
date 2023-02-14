import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


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
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3, adaptive=True, attention=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = num_subset#划分子集的方式，按最初的ST-GCN的第三种方式，一共被划分为3个子集，因此num_set=3。并且个st-gcn单元都有自己的权重参数用于训练参见下图
        num_jpts = A.shape[-1]

        self.conv_d = nn.ModuleList()#类似于pytho中的list类型，只是将一系列层装入列表，并没有实现forward()方法
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))#构建3个卷积层

        if adaptive:#自适应
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
            #复制A，torch.from_numpy,The returned tensor and `ndarray` share the same memory. Modifications to the tensor will be reflected in the `ndarray` and vice versa. The returned tensor is not resizable.
            #后半部分nn.Parameter，将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面
            self.alpha = nn.Parameter(torch.zeros(1))
            # self.beta = nn.Parameter(torch.ones(1))
            # nn.init.constant_(self.PA, 1e-6)
            # self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
            # self.A = self.PA
            self.conv_a = nn.ModuleList()
            self.conv_b = nn.ModuleList()
            for i in range(self.num_subset):
                self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
                self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            #上面的部分一共构建了3个卷积块conv_d，conv_a，conv_b，每个卷积块有3（num_subset）个卷积层
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.adaptive = adaptive

        if attention:#是否添加注意力机制
            # self.beta = nn.Parameter(torch.zeros(1))
            # self.gamma = nn.Parameter(torch.zeros(1))
            # unified attention
            # self.Attention = nn.Parameter(torch.ones(num_jpts))

            # temporal attention 时间维度注意力
            self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)#1d卷积
            nn.init.constant_(self.conv_ta.weight, 0)#初始化常量
            nn.init.constant_(self.conv_ta.bias, 0)

            # s attention 空间维度注意力
            ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts #ker_jpt始终为奇数
            pad = (ker_jpt - 1) // 2 #偶数填充
            self.conv_sa = nn.Conv1d(out_channels, 1, ker_jpt, padding=pad)
            nn.init.xavier_normal_(self.conv_sa.weight)
            nn.init.constant_(self.conv_sa.bias, 0)

            # channel attention 通道级注意力
            rr = 2
            self.fc1c = nn.Linear(out_channels, out_channels // rr)
            self.fc2c = nn.Linear(out_channels // rr, out_channels)
            nn.init.kaiming_normal_(self.fc1c.weight) #Xavier初始化方法
            nn.init.constant_(self.fc1c.bias, 0)
            nn.init.constant_(self.fc2c.weight, 0)
            nn.init.constant_(self.fc2c.bias, 0)

            # self.bn = nn.BatchNorm2d(out_channels)
            # bn_init(self.bn, 1)
        self.attention = attention

        if in_channels != out_channels:#如果输入通道与输出通道不相同，则调整至out_channel大小
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

#如何调用构建的模型
    def forward(self, x):
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            A = self.PA
            # A = A + self.PA
            for i in range(self.num_subset):
                A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)#N*C(e)T
                A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)#C(e)T*N
                A1 = self.tan(torch.matmul(A1, A2) / A1.size(-1))  #N*N
                A1 = A[i] + A1 * self.alpha
                A2 = x.view(N, C * T, V) #C(in)T*N
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z #...-> f(out)
        else:
            A = self.A.cuda(x.get_device()) * self.mask
            for i in range(self.num_subset):
                A1 = A[i]
                A2 = x.view(N, C * T, V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        if self.attention:
            # spatial attention
            se = y.mean(-2)  # N C V 归一化
            se1 = self.sigmoid(self.conv_sa(se))#计算注意力机制
            y = y * se1.unsqueeze(-2) + y
            # a1 = se1.unsqueeze(-2)

            # temporal attention
            se = y.mean(-1)
            se1 = self.sigmoid(self.conv_ta(se))
            y = y * se1.unsqueeze(-1) + y
            # a2 = se1.unsqueeze(-1)

            # channel attention
            se = y.mean(-1).mean(-1)#mean:cacluate the average
            se1 = self.relu(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))
            y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
            # a3 = se2.unsqueeze(-1).unsqueeze(-1)

            # unified attention
            # y = y * self.Attention + y
            # y = y + y * ((a2 + a3) / 2)
            # y = self.bn(y)
        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, attention=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive, attention=attention)#计算图卷积
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)#计算时间卷积
        self.relu = nn.ReLU(inplace=True)
        if attention:
            self.alpha = nn.Parameter(torch.zeros(1))
            self.beta = nn.Parameter(torch.ones(1))
            #temporal attention
            rt = 1
            self.conv_ta1 = nn.Conv1d(out_channels, out_channels//rt, 9, padding=4)
            self.bn = nn.BatchNorm2d(out_channels)
            bn_init(self.bn, 1)
            self.conv_ta2 = nn.Conv1d(out_channels, 1, 9, padding=4)
            nn.init.kaiming_normal_(self.conv_ta1.weight)
            nn.init.constant_(self.conv_ta1.bias, 0)
            nn.init.constant_(self.conv_ta2.weight, 0)
            nn.init.constant_(self.conv_ta2.bias, 0)

            rt = 4
            self.inter_c = out_channels // rt
            self.conv_ta1 = nn.Conv2d(out_channels, out_channels // rt, 1)
            self.conv_ta2 = nn.Conv2d(out_channels, out_channels // rt, 1)
            nn.init.constant_(self.conv_ta1.weight, 0)
            nn.init.constant_(self.conv_ta1.bias, 0)
            nn.init.constant_(self.conv_ta2.weight, 0)
            nn.init.constant_(self.conv_ta2.bias, 0)
            #s attention
            num_jpts = A.shape[-1]
            ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
            pad = (ker_jpt - 1) // 2
            self.conv_sa = nn.Conv1d(out_channels, 1, ker_jpt, padding=pad)
            nn.init.constant_(self.conv_sa.weight, 0)
            nn.init.constant_(self.conv_sa.bias, 0)

            #channel attention
            #rr = 16
            #self.fc1c = nn.Linear(out_channels, out_channels // rr)
            #self.fc2c = nn.Linear(out_channels // rr, out_channels)
            #nn.init.kaiming_normal_(self.fc1c.weight)
            #nn.init.constant_(self.fc1c.bias, 0)
            #nn.init.constant_(self.fc2c.weight, 0)
            #nn.init.constant_(self.fc2c.bias, 0)
            
            self.softmax = nn.Softmax(-2)
            self.sigmoid = nn.Sigmoid()
        self.attention = attention

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        if self.attention:
            y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))

            #spatial attention
            se = y.mean(-2)  # N C V
            se1 = self.sigmoid(self.conv_sa(se))
            y = y * se1.unsqueeze(-2) + y
            a1 = se1.unsqueeze(-2)

            #temporal attention
            se = y.mean(-1)  # N C T
            # se1 = self.relu(self.bn(self.conv_ta1(se)))
            se2 = self.sigmoid(self.conv_ta2(se))
            # y = y * se1.unsqueeze(-1) + y
            a2 = se2.unsqueeze(-1)

            se = y  # NCTV
            N, C, T, V = y.shape
            se1 = self.conv_ta1(se).permute(0, 2, 1, 3).contiguous().view(N, T, self.inter_c * V)  # NTCV
            se2 = self.conv_ta2(se).permute(0, 1, 3, 2).contiguous().view(N, self.inter_c * V, T)  # NCVT
            a2 = self.softmax(torch.matmul(se1, se2) / np.sqrt(se1.size(-1)))  # N T T
            y = torch.matmul(y.permute(0, 1, 3, 2).contiguous().view(N, C * V, T), a2) \
                    .view(N, C, V, T).permute(0, 1, 3, 2) * self.alpha + y

            #channel attention
            #se = y.mean(-1).mean(-1)
            #se1 = self.relu(self.fc1c(se))
            #se2 = self.sigmoid(self.fc2c(se1))
            # y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
            #a3 = se2.unsqueeze(-1).unsqueeze(-1)
            
            #y = y * ((a2 + a3) / 2) + y
            y = y * ((a1 + a2)  / 2) + y
            y = self.bn(y)
        else:
            y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, attention=True):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.num_class = num_class#动作类别数

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)#网络模型构建

        self.l1 = TCN_GCN_unit(3, 64, A, residual=False, adaptive=adaptive, attention=attention)
        self.l2 = TCN_GCN_unit(64, 64, A, adaptive=adaptive, attention=attention)
        self.l3 = TCN_GCN_unit(64, 64, A, adaptive=adaptive, attention=attention)
        self.l4 = TCN_GCN_unit(64, 64, A, adaptive=adaptive, attention=attention)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, adaptive=adaptive, attention=attention)
        self.l6 = TCN_GCN_unit(128, 128, A, adaptive=adaptive, attention=attention)
        self.l7 = TCN_GCN_unit(128, 128, A, adaptive=adaptive, attention=attention)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2, adaptive=adaptive, attention=attention)
        self.l9 = TCN_GCN_unit(256, 256, A, adaptive=adaptive, attention=attention)
        self.l10 = TCN_GCN_unit(256, 256, A, adaptive=adaptive, attention=attention)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        N, C, T, V, M = x.size()#对应输入数据格式，N=sample数量，C=（channel，3固定），T=帧，V=（关节数，固定25），M=（最大人数，固定2）


        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)#permute更换tensor维度的顺序；
        #view通常会使用contiguous，与内存有关，重新强制拷贝一份tensor
        #view顺序取数组重新组成目标形状
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)
