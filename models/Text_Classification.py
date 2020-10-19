# @Author : bamtercelboo
# @Datetime : 2018/9/14 8:43
# @File : Sequence_Label.py
# @Last Modify Time : 2018/9/14 8:43
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Sequence_Label.py
    FUNCTION : None
"""

import torch
import torch.nn as nn
import random
import numpy as np
import time
from models.BiLSTM import BiLSTM
from models.AttBiLSTM import ABiLSTM
from models.AttCNN import ACNN
from models.CNN import CNN
from models.CNN1 import CNN1
from models.Transformers import Transformer
from models.MultiCNN import CNN_MUI
from models.deepCNN import DCNN
from models.HighwayCNN import HighWay_CNN
from models.CBiLSTM import CNN_BiLSTM
from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class Text_Classification(nn.Module):
    """
        Sequence_Label
    """

    def __init__(self, config):
        super(Text_Classification, self).__init__()
        self.config = config
        # embed
        self.embed_num = config.embed_num
        self.embed_dim = config.embed_dim
        self.label_num = config.label_num
        self.paddingId = config.paddingId
        # dropout
        self.dropout_emb = config.dropout_emb
        self.dropout = config.dropout
        # lstm
        self.lstm_hiddens = config.lstm_hiddens
        self.lstm_layers = config.lstm_layers
        # pre train
        self.pretrained_embed = config.pretrained_embed
        self.pretrained_weight = config.pretrained_weight
        # cnn param
        self.wide_conv = config.wide_conv
        self.conv_filter_sizes = self._conv_filter(config.conv_filter_sizes)
        self.conv_filter_nums = config.conv_filter_nums
        self.use_cuda = True
        if config.device == cpu_device:
            self.use_cuda = False
        # transformer
        self.num_head = 5
        self.hidden = 1024
        self.last_hidden = 512
        self.num_encoder = 2
        self.pad_size = 32

        if self.config.model_bilstm:
            self.model = BiLSTM(embed_num=self.embed_num, embed_dim=self.embed_dim, label_num=self.label_num,
                                 paddingId=self.paddingId, dropout_emb=self.dropout_emb, dropout=self.dropout,
                                 lstm_hiddens=self.lstm_hiddens, lstm_layers=self.lstm_layers,
                                 pretrained_embed=self.pretrained_embed, pretrained_weight=self.pretrained_weight,
                                 use_cuda=self.use_cuda)
        elif self.config.model_attbilstm:
            self.model = ABiLSTM(embed_num=self.embed_num, embed_dim=self.embed_dim, label_num=self.label_num,
                                 paddingId=self.paddingId, dropout_emb=self.dropout_emb, dropout=self.dropout,
                                 lstm_hiddens=self.lstm_hiddens, lstm_layers=self.lstm_layers,
                                 pretrained_embed=self.pretrained_embed, pretrained_weight=self.pretrained_weight,
                                 use_cuda=self.use_cuda)
        elif self.config.model_cnn:
            self.model = CNN(embed_num=self.embed_num, embed_dim=self.embed_dim, label_num=self.label_num,
                             paddingId=self.paddingId, dropout_emb=self.dropout_emb, dropout=self.dropout,
                             conv_filter_nums=self.conv_filter_nums, conv_filter_sizes=self.conv_filter_sizes,
                             wide_conv=self.wide_conv,
                             pretrained_embed=self.pretrained_embed, pretrained_weight=self.pretrained_weight,
                             use_cuda=self.use_cuda)
        elif self.config.model_cnn1:
            self.model = CNN1(embed_num=self.embed_num, embed_dim=self.embed_dim, label_num=self.label_num,
                             paddingId=self.paddingId, dropout_emb=self.dropout_emb, dropout=self.dropout,
                             conv_filter_nums=self.conv_filter_nums, conv_filter_sizes=self.conv_filter_sizes,
                             wide_conv=self.wide_conv,
                             pretrained_embed=self.pretrained_embed, pretrained_weight=self.pretrained_weight,
                             use_cuda=self.use_cuda)
        elif self.config.model_transformer:
            self.model = Transformer(embed_num=self.embed_num, embed_dim=self.embed_dim, label_num=self.label_num,
                             paddingId=self.paddingId, dropout_emb=self.dropout_emb, dropout=self.dropout,
                             num_head=self.num_head, hidden=self.hidden, last_hidden =self.last_hidden,
                             num_encoder=self.num_encoder, pad_size=self.pad_size,
                             pretrained_embed=self.pretrained_embed, pretrained_weight=self.pretrained_weight,
                             use_cuda=self.use_cuda)
        elif self.config.model_attcnn:
            self.model = ACNN(embed_num=self.embed_num, embed_dim=self.embed_dim, label_num=self.label_num,
                              paddingId=self.paddingId, dropout_emb=self.dropout_emb, dropout=self.dropout,
                              conv_filter_nums=self.conv_filter_nums, conv_filter_sizes=self.conv_filter_sizes,
                              wide_conv=self.wide_conv,
                              pretrained_embed=self.pretrained_embed, pretrained_weight=self.pretrained_weight,
                              use_cuda=self.use_cuda)
        elif self.config.model_deepcnn:
            self.model = DCNN(embed_num=self.embed_num, embed_dim=self.embed_dim, label_num=self.label_num,
                              paddingId=self.paddingId, dropout_emb=self.dropout_emb, dropout=self.dropout,
                              conv_filter_nums=self.conv_filter_nums, conv_filter_sizes=self.conv_filter_sizes,
                              wide_conv=self.wide_conv,
                              pretrained_embed=self.pretrained_embed, pretrained_weight=self.pretrained_weight,
                              use_cuda=self.use_cuda)
        elif self.config.model_highwaycnn:
            self.model = HighWay_CNN(embed_num=self.embed_num, embed_dim=self.embed_dim, label_num=self.label_num,
                                     paddingId=self.paddingId, dropout_emb=self.dropout_emb, dropout=self.dropout,
                                     conv_filter_nums=self.conv_filter_nums, conv_filter_sizes=self.conv_filter_sizes,
                                     wide_conv=self.wide_conv,
                                     pretrained_embed=self.pretrained_embed, pretrained_weight=self.pretrained_weight,
                                     use_cuda=self.use_cuda)
        elif self.config.model_multicnn:
            self.model = CNN_MUI(embed_num=self.embed_num, embed_dim=self.embed_dim, label_num=self.label_num,
                              paddingId=self.paddingId, dropout_emb=self.dropout_emb, dropout=self.dropout,
                              conv_filter_nums=self.conv_filter_nums, conv_filter_sizes=self.conv_filter_sizes,
                              wide_conv=self.wide_conv,
                              pretrained_embed=self.pretrained_embed, pretrained_weight=self.pretrained_weight,
                              use_cuda=self.use_cuda)
        elif self.config.model_cnn_bilstm:
            self.model = CNN_BiLSTM(embed_num=self.embed_num, embed_dim=self.embed_dim, label_num=self.label_num,
                                    paddingId=self.paddingId, dropout_emb=self.dropout_emb, dropout=self.dropout,
                                    conv_filter_nums=self.conv_filter_nums, conv_filter_sizes=self.conv_filter_sizes,
                                    lstm_hiddens=self.lstm_hiddens, lstm_layers=self.lstm_layers,
                                    wide_conv=self.wide_conv,
                                    pretrained_embed=self.pretrained_embed, pretrained_weight=self.pretrained_weight,
                                    use_cuda=self.use_cuda)
    @staticmethod
    def _conv_filter(str_list):
        """
        :param str_list:
        :return:
        """
        int_list = []
        str_list = str_list.split(",")
        for str in str_list:
            int_list.append(int(str))
        return int_list

    def forward(self, word, sentence_length, train=False):
        """
        :param char:
        :param word:
        :param sentence_length:
        :param train:
        :return:
        """

        model_output = self.model(word, sentence_length)
        return model_output


