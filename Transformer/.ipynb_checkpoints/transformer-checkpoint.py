import torch.nn as nn
import model as model
import decoder as decoder
import encoder as encoder
import projection_layer as projection_layer

class Transformer(nn.Module):
    def __init__(self,
                 encoder :encoder.Encoder,
                 decoder :decoder.Decoder,
                 src_embed :model.InputEmbedding,
                 tgt_embed :model.InputEmbedding,
                 src_pos :model.PositionalEncoding,
                 tgt_pos :model.PositionalEncoding,
                 projection_layer :projection_layer.ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)

def build_transformer(src_vocab_size :int, tgt_vocab_size :int, src_seq_len :int, tgt_seq_len :int,
                       d_model :int=512, N :int=6, h :int=8, dropout :float=0.1,
                       d_ff :int=2048):
    #embedding layer
    src_embed = model.InputEmbedding(d_model, src_vocab_size)
    tgt_embed = model.InputEmbedding(d_model, tgt_seq_len)

    #posn encoding layer
    src_pos = model.PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = model.PositionalEncoding(d_model, tgt_seq_len, dropout)

    #encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attn_block = model.MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = model.FeedForwardBlock(d_model, d_ff, dropout)

        encoder_block = model.EncoderBlock(encoder_self_attn_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attn_block = model.MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attn_block = model.MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = model.FeedForwardBlock(d_model, d_ff, dropout)

        decoder_block = model.DecoderBlock(decoder_self_attn_block, decoder_cross_attn_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    #Encoder and Decoder
    enc = encoder.Encoder(nn.ModuleList(encoder_blocks))
    dec = decoder.Decoder(nn.ModuleList(decoder_blocks))

    proj_layer = projection_layer.ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(enc, dec,
                               src_embed, tgt_embed,
                                src_pos, tgt_pos, proj_layer)

    #Initialize the param with Xavier's

    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)

    return transformer


    
        