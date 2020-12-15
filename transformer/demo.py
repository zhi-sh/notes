# -*- coding: utf-8 -*-
# @DateTime :2020/12/14 下午9:29
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import numpy as np
import torch
from torch.autograd import Variable

from tools import Batch, LabelSmoothing, NoamOpt, SimpleLossCompute, run_epoch
from transformer import make_transformer


def data_gen(V, batch, nbatches):
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        d = Batch(src, tgt, 0)
        yield d


if __name__ == '__main__':
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_transformer(V, V, N=2)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(10):
        model.train()
        run_epoch(data_gen(V, 30, 20), model, SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(run_epoch(data_gen(V, 30, 5), model, SimpleLossCompute(model.generator, criterion, None)).item())
