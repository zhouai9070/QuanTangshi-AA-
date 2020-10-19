# @Author : bamtercelboo
# @Datetime : 2018/10/15 9:52
# @File : CNN.py
# @Last Modify Time : 2018/10/15 9:52
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  CNN.py
    FUNCTION : None
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
from DataUtils.Common import *
from models.initialize import *
from models.modelHelp import prepare_pack_padded_sequence
torch.manual_seed(seed_num)
random.seed(seed_num)


class DCNN(nn.Module):
    """
        BiLSTM
    """

    def __init__(self, **kwargs):
        super(DCNN, self).__init__()
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

        # cnn
        if self.wide_conv:
            print("Using Wide Convolution")
            self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=D, kernel_size=(K, D), stride=(1, 1),
                                   padding=(K // 2, 0), bias=False) for K in kernel_sizes]
            self.convs2 = [nn.Conv2d(in_channels=Ci, out_channels=kernel_nums, kernel_size=(K, D), stride=(1, 1),
                                     padding=(K // 2, 0), bias=False) for K in kernel_sizes]
        else:
            print("Using Narrow Convolution")
            self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=D, kernel_size=(K, D), bias=True) for K in
                           kernel_sizes]
            self.convs2 = [nn.Conv2d(in_channels=Ci, out_channels=kernel_nums, kernel_size=(K, D), bias=True) for K in
                           kernel_sizes]
            print(self.convs1)
            print(self.convs2)

        for conv in self.convs1:
            if self.use_cuda:
                conv.cuda()
        for conv in self.convs2:
            if self.use_cuda:
                conv.cuda()
        #self.dropout = nn.Dropout(self.dropout)
        in_fea = len(kernel_sizes) * kernel_nums
        self.fc1 = nn.Linear(in_features=in_fea, out_features=in_fea // 2, bias=True)
        self.fc2 = nn.Linear(in_features=in_fea // 2, out_features=C, bias=True)
        init_linear_weight_bias(self.fc2)

    def forward(self, word, sentence_length):
        """
        :param word:
        :param sentence_length:
        :return:
        """
        x = self.embed(word)  # (N,W,D)
        x = self.dropout_embed(x)
        x = x.unsqueeze(1)  # (N,Ci,W,D)
        x = [torch.transpose(F.relu(conv(x)).squeeze(3), 1, 2) for conv in self.convs1]
        #x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
        x = [F.relu(conv(x.unsqueeze(1))).squeeze(3) for (conv, x) in zip(self.convs2, x)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N,len(Ks)*Co)
        x = self.fc1(F.relu(x))
        logit = self.fc2(F.relu(x))
        return logit