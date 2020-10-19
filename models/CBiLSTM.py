# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : model_CNN_BiLSTM.py
# @Last Modify Time : 2018/07/19 22:35
# @Contact : bamtercelboo@{gmail.com, 163.com}

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from DataUtils.Common import seed_num
from models.initialize import init_embedding

torch.manual_seed(seed_num)
random.seed(seed_num)


"""
    Neural Network: CNN_BiLSTM
    Detail: the input cross cnn model and LSTM model independly, then the result of both concat
"""


class CNN_BiLSTM(nn.Module):

    def __init__(self, **kwargs):
        super(CNN_BiLSTM, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])
        self.hidden_dim = self.lstm_hiddens
        self.num_layers = self.lstm_layers
        V = self.embed_num
        D = self.embed_dim
        C = self.label_num
        self.C = C
        Ci = 1
        kernel_nums = self.conv_filter_nums
        kernel_sizes = self.conv_filter_sizes
        paddingId = self.paddingId
        self.embed = nn.Embedding(V, D, padding_idx= paddingId)
        # pretrained  embedding
        if self.pretrained_embed:
            self.embed.weight.data.copy_(self.pretrained_weight)
        else:
            init_embedding(self.embed.weight)

        # CNN
        if self.wide_conv:
            print("Using Wide Convolution")
            self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=kernel_nums, kernel_size=(K, D), stride=(1, 1),
                                     padding=(K // 2, 0), bias=False) for K in kernel_sizes]
        else:
            print("Using Narrow Convolution")
            self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=kernel_nums, kernel_size=(K, D), bias=True) for K in
                           kernel_sizes]
            print(self.convs1)
        # for cnn cuda
        for conv in self.convs1:
            if self.use_cuda:
                conv.cuda()

        # BiLSTM
        self.bilstm = nn.LSTM(input_size=D, hidden_size=self.lstm_hiddens, num_layers=self.lstm_layers,
                              bidirectional=True, batch_first=True, bias=True)

        # linear
        L = len(kernel_sizes) * kernel_nums + self.hidden_dim * 2
        self.hidden2label1 = nn.Linear(L, L // 2)
        self.hidden2label2 = nn.Linear(L // 2, C)

        # dropout
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, word, sentence_length):
        embed = self.embed(word)

        # CNN
        cnn_x = embed
        cnn_x = torch.transpose(cnn_x, 0, 1)
        cnn_x = cnn_x.unsqueeze(1)
        cnn_x = [conv(cnn_x).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        cnn_x = [F.tanh(F.max_pool1d(i, i.size(2)).squeeze(2)) for i in cnn_x]  # [(N,Co), ...]*len(Ks)
        cnn_x = torch.cat(cnn_x, 1)
        cnn_x = self.dropout(cnn_x)

        # BiLSTM
        bilstm_x = embed#.view(len(word), embed.size(1),-1)
        bilstm_out, _ = self.bilstm(bilstm_x)
        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 1, 2)
        bilstm_out = bilstm_out.squeeze(2)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        bilstm_out = F.tanh(bilstm_out)

        # CNN and BiLSTM CAT
        cnn_x = torch.transpose(cnn_x, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        cnn_bilstm_out = torch.cat((cnn_x, bilstm_out), 0)
        cnn_bilstm_out = torch.transpose(cnn_bilstm_out, 0, 1)

        # linear
        cnn_bilstm_out = self.hidden2label1(F.tanh(cnn_bilstm_out))
        cnn_bilstm_out = self.hidden2label2(F.tanh(cnn_bilstm_out))

        # output
        logit = cnn_bilstm_out
        return logit
