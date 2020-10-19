# @Author : bamtercelboo
# @Datetime : 2018/1/30 15:58
# @File : DataConll2003_Loader.py
# @Last Modify Time : 2018/1/30 15:58
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :
    FUNCTION :
"""
import sys
import os
import re
import random
import torch
from Dataloader.Instance import Instance
from xpinyin import Pinyin
from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)
pinyin = Pinyin()

# create stopwords list
def stopwordslist():
    stopwords = [line.strip() for line in open('/Users/macbook/Desktop/stopword.txt',encoding='UTF-8').readlines()]
    return stopwords
# delete stopwords
def del_stopword(sentence):
    stopwords = stopwordslist()
    outstr = ''
    for word in sentence:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += ""
    return outstr

class DataLoaderHelp(object):
    """
    DataLoaderHelp
    """

    @staticmethod
    def _clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    @staticmethod
    def _normalize_word(word):
        """
        :param word:
        :return:
        """
        new_word = ""
        for char in word:
            if char.isdigit():
                new_word += '0'
            else:
                new_word += char
        return new_word

    @staticmethod
    def _sort(insts):
        """
        :param insts:
        :return:
        """
        sorted_insts = []
        sorted_dict = {}
        for id_inst, inst in enumerate(insts):
            sorted_dict[id_inst] = inst.words_size
        dict = sorted(sorted_dict.items(), key=lambda d: d[1], reverse=True)
        for key, value in dict:
            sorted_insts.append(insts[key])
        print("Sort Finished.")
        return sorted_insts


class DataLoader(DataLoaderHelp):
    """
    DataLoader
    """
    def __init__(self, path, shuffle, config):
        """
        :param path: data path list
        :param shuffle:  shuffle bool
        :param config:  config
        """
        #
        print("Loading Data......")
        self.data_list = []
        self.max_count = config.max_count
        self.path = path
        self.shuffle = shuffle

    def dataLoader(self):
        """
        :return:
        """
        path = self.path
        shuffle = self.shuffle
        assert isinstance(path, list), "Path Must Be In List"
        print("Data Path {}".format(path))
        for id_data in range(len(path)):
            print("Loading Data Form {}".format(path[id_data]))
            insts = self._Load_Each_Data(path=path[id_data], shuffle=shuffle)
            if shuffle is True and id_data == 0:
                print("shuffle train data......")
                random.shuffle(insts)
            self.data_list.append(insts)
        # return train/dev/test data
        if len(self.data_list) == 3:
            # 获取测试集的原始数据
            '''
            print('$$$$$$$$$$$$$$$$$')
            for k in self.data_list[2]:
                print(k.label_index)
            print('$$$$$$$$$$$$$$$$$')
            '''
            return self.data_list[0], self.data_list[1], self.data_list[2]
        elif len(self.data_list) == 2:
            return self.data_list[0], self.data_list[1]

    def _Load_Each_Data(self, path=None, shuffle=False):
        """
        :param path:
        :param shuffle:
        :return:
        """
        #pinyin1 = Pinyin()
        assert path is not None, "The Data Path Is Not Allow Empty."
        insts = []
        now_lines = 0
        with open(path, encoding="UTF-8") as f:
            inst = Instance()
            data = [line.strip().split() for line in f]
            for line in data:
                #line = line.strip()
                inst = Instance()
                #line = line.split()
                label = line[0]
                #word = line[2:-1]   # No LDA
                word = line[2:]    # Have LDA
                #pinyin = pinyin1.get_pinyin(word," ", tone_marks= None)
                #word = " ".join(line[1:])
                inst.words = word
                inst.labels.append(label)
                inst.words_size = len(inst.words)
                #inst.pinyin = pinyin
                insts.append(inst)
                if len(insts) == self.max_count:
                    break
        '''
        得到原始的数据，可只用于test中
        或者将它写入txt 对比
        print('$$$$$$$$$$$$$$$$$')
        for k in insts:
            print(k.labels, '****', k.words)
        print('################')
        '''
        return insts
