# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : model_CNN_MUI.py
# @Last Modify Time : 2018/07/19 22:35
# @Contact : bamtercelboo@{gmail.com, 163.com}

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from pypinyin import pinyin
import torch.nn.init as init

from DataUtils import Embed
from DataUtils.Common import seed_num
from models.initialize import *
torch.manual_seed(seed_num)
random.seed(seed_num)

"""
Description:
    the model is a mulit-channel CNNS model, 
    the model use two external word embedding, and then,
    one of word embedding built from train/dev/test dataset,
    and it be used to no-fine-tune,other one built from only
    train dataset,and be used to fine-tune.

    my idea,even if the word embedding built from train/dev/test dataset, 
    whether can use fine-tune, in others words, whether can fine-tune with
    two external word embedding.
"""


class CNN_MUI(nn.Module):

    def __init__(self, **kwargs):
        super(CNN_MUI, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

        V = self.embed_num
        #V_mui = self.embed_num_mui
        D = self.embed_dim
        C = self.label_num
        Ci = 2
        kernel_nums = self.conv_filter_nums
        kernel_sizes = self.conv_filter_sizes
        paddingId = self.paddingId

        self.embed = nn.Embedding(V, D, padding_idx=paddingId)
        self.embed1 = nn.Embedding(V, D, padding_idx=paddingId)

        #self.preembed = Embed()
        if self.pretrained_embed:
            self.embed.weight.data.copy_(self.pretrained_weight)
        #else:
            init_embedding(self.embed1.weight)

        if self.wide_conv is True:
            print("using wide convolution")
            self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=kernel_nums, kernel_size=(K, D), stride=(1, 1),
                                     padding=(K // 2, 0), bias=True) for K in kernel_sizes]
        else:
            print("using narrow convolution")
            self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=kernel_nums, kernel_size=(K, D), bias=True) for K in kernel_sizes]
        print(self.convs1)

        self.dropout_embed = nn.Dropout(self.dropout_emb)
        self.dropout = nn.Dropout(self.dropout)

        # for cnn cuda
        for conv in self.convs1:
            if self.use_cuda:
                conv.cuda()

        in_fea = len(kernel_sizes) * kernel_nums
        self.fc1 = nn.Linear(in_features=in_fea, out_features=in_fea // 2, bias=True)
        self.fc2 = nn.Linear(in_features=in_fea // 2, out_features=C, bias=True)



    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, word, sentence_length):
        x_no_static = self.embed1(word)
        x_static = self.embed(word)
        x = torch.stack([x_static, x_no_static], 1)
        x = self.dropout(x)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N,len(Ks)*Co)
        x = self.fc1(x)
        logit = self.fc2(F.relu(x))
        return logit
