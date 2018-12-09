# -*- coding: utf-8 -*-
# @author: gcg
# 句子级文本分类的数据处理
# loadw2v函数：返回单词-id的字典和词向量矩阵
# convert2id 根据一个最大词数，将文档矩阵里的词全转换成id。
# 对长文档进行截断，短文档进行补0。所有没有在词典里出现的单词映射到一个UNK。UNK是一个全0向量。
import numpy as np


def load_w2v(w2v_file='./data/sst-Google-vectors.txt'):
    fp = open(w2v_file)
    words, embedding_dim = map(int, fp.readline().split())
    w2v = []
    # [0,0,...,0] represent absent words
    w2v.append([0.] * embedding_dim)
    word_dict = dict()
    cnt = 0
    for line in fp:
        cnt += 1
        line = line.split()
        word_dict[line[0]] = cnt
        w2v.append([float(v) for v in line[1:]])
    w2v = np.array(w2v, dtype=np.float32)
    return word_dict, w2v


def loadw2v(file_name='./data/sst-Google-vectors.txt'):
    '''
    第一个单词是UNK，用来表示词典外的单词，初始化为零向量
    :param file_name: 存放词向量的文件名，第一行2个数字是（包含词数目，词向量维度）
    :return: word2id:关键字是单词，值是id的字典。embedding：词向量矩阵，类型是列表
    '''
    with open(file_name, 'r', encoding='utf-8') as fp:
        content = fp.read().splitlines()
    word_number, embed_dim = map(int, content[0].split())
    embedding = [[0.] * embed_dim]  # 初始化词向量矩阵
    word_list = ['UNK']
    word_index = [0]
    for line in range(1, len(content)):
        temp_line = content[line].split()
        word_list.append(temp_line[0])
        word_index.append(line)
        embedding.append(list(map(float, temp_line[1:-1])))
    word2id = dict(zip(word_list, word_index))
    return word2id, np.asarray(embedding)


def convert_word2id(word, w2id, maxword):
    '''
    :param word:  一个列表，元素是token，每一行是一句话。
    :param w2id: 关键字是单词，值是id的字典
    :param maxword: 一句话最多包含的字符数目
    :return: word_index一个长度等于列表长度，宽度等于maxword的列表，sen_len:一个长度为句子数的列表，存放每条句子有效词数目
    '''
    # word_index：numpy数组，形状为[len(word),maxword]
    word_index = np.zeros([len(word), maxword], dtype=int)
    sen_len = [0] * len(word)
    count = 0  # 计数有效单词个数
    for i in range(len(word)):
        for j in range(len(word[i])):
            if j < maxword:
                if word[i][j] in w2id.keys():
                    word_index[i][j] = w2id[word[i][j]]
                else:
                    word_index[i][j] = 0
        if j < maxword:
            sen_len[i] = j+1
        else:
            sen_len[i] = j
    return word_index, sen_len

# features, labels = create_dataset()
# w2id, _ = loadw2v()
# w_index, l = convert_word2id(features, w2id, 40)
# print(len(features), w_index.shape)
# print(w_index[1000], '\n', features[1000], l[1000])
