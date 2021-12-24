import torch
import torch.nn as nn
from operations import *
import torch.nn.functional as F
from torch.autograd import Variable
from utils_original import drop_path
from train_model.aspp import Original_ASPP
from train_model.decoder import Decoder

class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvGN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvGN(C_prev, C, 1, 1, 0)    
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class NetworkCityscapes(nn.Module):

    def __init__(self, C, num_classes, layers, crop_size, genotype):
        super(NetworkCityscapes, self).__init__()
        self._layers = layers
        self._num_classes = num_classes
        self._crop_size = crop_size

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, C // 2),
            nn.ReLU(inplace=True),
        )
        self.stem1 = nn.Sequential(
            nn.Conv2d(C // 2, C // 2, 3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, C // 2),
            nn.ReLU(inplace=True),
        )
        self.stem2 = nn.Sequential(
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, C),
            nn.ReLU(inplace=True),
        )

        C_prev_prev, C_prev, C_curr = C // 2, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(self._layers):
            if i in [self._layers // 3, 2 * self._layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

        self.ASPP = Original_ASPP(C_prev, 256)
        self.decoder = Decoder(self._num_classes, low_level_inplanes= C * 4)

       self._load_pretrained_model()

    def forward(self, input):
        s = self.stem0(input)
        s0 = self.stem1(s)
        s1 = self.stem2(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 1:
                low_level_feature = s1
        high_level_feature = self.ASPP(s1)
        decoder_output = self.decoder(high_level_feature, low_level_feature)
        final_output = F.interpolate(decoder_output, (self._crop_size, self._crop_size), None, 'bilinear', True)
        return final_output

    def _load_pretrained_model(self):
        checkpoint = torch.load('/home/zhou1/GNAS2-cityscapes/Pretrain-ImageNet/checkpoints-20200107-232520/model_best.pth.tar')
        pretrain_dict = checkpoint['state_dict']
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)