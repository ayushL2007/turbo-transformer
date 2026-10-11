import torch.nn as nn
import torch
import model as model

class ProjectionLayer(nn.Module):
    def __init__(self, d_model :int, vocab_size :int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        #(_nbatch, _nseq, d_model)->(_nbatch, _nseq, _nvocab)
        return torch.log_softmax(self.proj(x), dim=-1)
