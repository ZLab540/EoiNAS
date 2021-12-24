import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


class MixedOp(nn.Module):

    def __init__(self, C, stride, switch, p):
        super(MixedOp, self).__init__()
        self.m_ops = nn.ModuleList()
        self.p = p
        self.switch = switch
        for i in range(len(self.switch)):
            if self.switch[i]:
                primitive = PRIMITIVES[i]
                op = OPS[primitive](C, stride, False)
                if 'pool' in primitive:
                    op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
                if isinstance(op, Identity) and p > 0:
                    op = nn.Sequential(op, nn.Dropout(self.p))
                self.m_ops.append(op)
                
    def update_p(self):
        for op in self.m_ops:
            if isinstance(op, nn.Sequential):
                if isinstance(op[0], Identity):
                    op[1].p = self.p

    def update_switch(self, index):
        del self.m_ops[index]


    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self.m_ops))


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, switches, p):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.p = p
        self.switches = switches
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self.cell_ops = nn.ModuleList()
        switch_count = 0
        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, switch=switches[switch_count], p=self.p)
                self.cell_ops.append(op)
                switch_count = switch_count + 1
    
    def update_p(self):
        for op in self.cell_ops:
            op.p = self.p
            op.update_p()

    def update_switches(self,indexes):
        count = 0
        for op in self.cell_ops:
            op.switch = self.switches[count]
            op.update_switch(indexes[count])
            count = count +1


    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self.cell_ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, num_classes, layers, criterion, steps=4, multiplier=4, crop_size=None, switches_normal=[], switches_reduce=[], p=0.0):
        super(Network, self).__init__()
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self._crop_size = crop_size
        self.p = p
        self.switches_normal = switches_normal
        self.switches_reduce = switches_reduce
        self.cells = nn.ModuleList()
        self._initialize_alphas()

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, 20, 3, stride=2, padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU()
        )
        self.stem1 = nn.Sequential(
            nn.Conv2d(20, 20, 3, padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU()
        )
        self.stem2 = nn.Sequential(
            nn.Conv2d(20, 40, 3, stride=2, padding=1),
            nn.BatchNorm2d(40),
            nn.ReLU()
        )

        C_prev_prev = 20
        C_prev = 40
        C_curr = 40

        reduction_prev = True
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
                cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, switches_reduce, self.p)
            else:
                reduction = False
                cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, switches_normal, self.p)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.ASPP = ASPP(C_prev, self._num_classes)


    def forward(self, input, temperature):
        stem = self.stem0(input)
        s0 = self.stem1(stem)
        s1 = self.stem2(s0)
        weights_reduce = F.gumbel_softmax(self.alphas_reduce, tau=temperature, hard=False, eps=1e-10)
        weights_normal = F.gumbel_softmax(self.alphas_normal, tau=temperature, hard=False, eps=1e-10)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = weights_reduce
            else:
                weights = weights_normal
            s0, s1 = s1, cell(s0, s1, weights)

        encoder_feature = self.ASPP(s1)
        aspp_result = F.interpolate(encoder_feature, (self._crop_size, self._crop_size), None, 'bilinear', True)
        return aspp_result

    def update_p(self):
        for cell in self.cells:
            cell.p = self.p
            cell.update_p()

    def update_switches(self, indexes_normal, indexes_reduce):
        for cell in self.cells:
            if cell.reduction:
                cell.switches = self.switches_reduce
                cell.update_switches(indexes_reduce)
            else:
                cell.switches = self.switches_normal
                cell.update_switches(indexes_normal)

    def update_arch_parameters(self, w_drop_normal, w_drop_reduce, column_len):
        w_keep_normal=[]
        w_keep_reduce=[]

        for i in range(14):
            for j in range(column_len + 1):
                 if j!=w_drop_normal[i]:
                     w_keep_normal.append(j)
            with torch.no_grad():
                 for k in range(column_len):
                    self.alphas_normal[i][k]=self.alphas_normal[i][w_keep_normal[k]]
                    self.alphas_normal.grad[i][k] = self.alphas_normal.grad[i][w_keep_normal[k]]
        for i in range(14):
            for j in range(column_len + 1):
                if j!=w_drop_reduce[i]:
                    w_keep_reduce.append(j)
            with torch.no_grad():
                for k in range(column_len):
                    self.alphas_reduce[i][k]=self.alphas_reduce[i][w_keep_reduce[k]]
                    self.alphas_reduce.grad[i][k] = self.alphas_reduce.grad[i][w_keep_reduce[k]]
        with torch.no_grad():
            self.alphas_normal.data = self.alphas_normal[:, 0:column_len]
            self.alphas_reduce.data = self.alphas_reduce[:, 0:column_len]
            self.alphas_normal.grad.data = self.alphas_normal.grad[:, 0:column_len]
            self.alphas_reduce.grad.data = self.alphas_reduce.grad[:, 0:column_len]

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target) 

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2+i))
        self.alphas_normal = nn.Parameter(torch.FloatTensor(1e-3 *np.random.randn(k, 8)))
        self.alphas_reduce = nn.Parameter(torch.FloatTensor(1e-3*np.random.randn(k, 8)))
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters


