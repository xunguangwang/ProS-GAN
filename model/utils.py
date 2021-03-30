import torch
from torch.autograd import Variable


def set_input_images(_input):
    _input = _input.cuda()
    _input = 2 * _input - 1
    return _input


def CalcSim(batch_label, train_label):
    S = (batch_label.mm(train_label.t()) > 0).float()
    return S


def log_trick(x):
    lt = torch.log(1 + torch.exp(-torch.abs(x))) + torch.max(
        x, Variable(torch.FloatTensor([0.]).cuda()))
    return lt
