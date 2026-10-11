import model as model
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, layers :nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = model.LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)

