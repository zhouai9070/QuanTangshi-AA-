# @Author : bamtercelboo
# @Datetime : 2018/10/15 9:52
# @File : CNN.py
# @Last Modify Time : 2018/10/15 9:52
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  CNN.py
    FUNCTION : None
"""
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
from DataUtils.Common import *
from models.initialize import *
from models.modelHelp import prepare_pack_padded_sequence
#from transformer.sublayers import MultiHeadAttention
from models import muti_hands
#from Utils import PositionwiseFeedForward,MultiHeadAttention
import torch.nn as nn
from .muti_hands import multihead_attention, feedforward

torch.manual_seed(seed_num)
random.seed(seed_num)

class CNN1(nn.Module):
    """
        BiLSTM
    """

    def __init__(self, **kwargs):
        super(CNN1, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

        V = self.embed_num
        D = self.embed_dim
        C = self.label_num
        Ci = 1
        kernel_nums = self.conv_filter_nums
        kernel_sizes = self.conv_filter_sizes
        paddingId = self.paddingId

        self.embed = nn.Embedding(V, D, padding_idx=paddingId)

        if self.pretrained_embed:
            self.embed.weight.data.copy_(self.pretrained_weight)
        else:
            init_embedding(self.embed.weight)

        self.dropout_embed = nn.Dropout(self.dropout_emb)
        self.dropout = nn.Dropout(self.dropout)
        self.feed = feedforward(in_channels=300)
        self.multi_att = multihead_attention(num_units=300)
        # cnn
        if self.wide_conv:
            print("Using Wide Convolution")
            self.conv = [nn.Conv2d(in_channels=Ci, out_channels=kernel_nums, kernel_size=(K, D), stride=(1, 1),
                                   padding=(K // 2, 0), bias=False) for K in kernel_sizes]
        else:
            print("Using Narrow Convolution")
            self.conv = [nn.Conv2d(in_channels=Ci, out_channels=kernel_nums, kernel_size=(K, D), bias=True) for K in kernel_sizes]
        print(self.conv)
        for conv in self.conv:
            if self.use_cuda:
                conv.cuda()

        in_fea = len(kernel_sizes) * kernel_nums
        self.linear = nn.Linear(in_features=in_fea, out_features=C, bias=True)
        init_linear_weight_bias(self.linear)

    def forward(self, word, sentence_length):
        """
        :param word:
        :param sentence_length:
        :return:
        """
        x = self.embed(word)  # (N,W,D)
        x = self.multi_att(x, x, x)
        x = self.feed(x)
        x = self.dropout_embed(x)
        x = x.unsqueeze(1)  # (N,Ci,W,D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv] #[(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        logit = self.linear(x)
        #logit = sm(logit)
        #print(logit)
        return logit