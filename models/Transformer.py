import torch
import torch.nn as nn

from models.Multihead import MultiHeadAttention
from models.Position import PositionalWiseFeedForward
from models.PositionEncoding import PositionalEncoding
from models.ScaleDotAttention import padding_mask


class EncoderLayer(nn.Module):

    def __init__(self, model_dim=300, num_heads=8, ffn_dim=1200, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):

        # self attention
        context, attention = self.attention(inputs, inputs, inputs, padding_mask)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention


class Encoder(nn.Module):


    def __init__(self,
               vocab_size,
               max_seq_len=4272,
               num_layers=1,
               model_dim=300,
               num_heads=8,
               ffn_dim=300,
               dropout=0.25):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
          [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
           range(num_layers)])

        self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        #self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, inputs):
        output = self.seq_embedding(inputs)
        #output += self.pos_embedding(inputs_len)

        self_attention_mask = padding_mask(inputs, inputs)

        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)

        return output, attentions
