# -*- coding: utf-8 -*-
# @author: gcg
import tensorflow as tf
from util.data_helpers import *
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
'''
dropout正则化在很多时候是重要的。
注意测试的时候不要
'''
sys.path.append("..")
# 模型参数
BATCH_SIZE = 200  # 训练集有8544条 验证集1101条数据 测试集2210条
learning_rate = 0.001
max_word = 200  # 一句话最多单词数，超出的单词进行截断。
HIDDEN_SIZE = 300  # lstm隐层单元数目
nclass = 5  # 类别数
skip_step = 100
test_step = 43
max_iters = 1000
L2_reg = 0.0001


def create_dataset(file_neme, word2index):
    '''
    :param file_neme: 数据集文件,可以是验证集、测试集、训练集，因为他们的格式是一样的。
    :return: features:numpy数组，存放所有句子，labels：列表，返回标签
    '''
    features = []
    labels = []
    with open(file_neme, 'r', encoding='utf-8') as fp:
        content = fp.read().splitlines()
    content = [temp.split('\t\t') for temp in content]
    for line in content:
        labels.append(int(line[0]))
        features.append(line[1].split())
    features = convert_word2id(features, word2index, max_word)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.shuffle(100)
    return dataset


w2id, embedding = load_w2v()  # 参数是词向量文件名，有默认值
validation_dataset = create_dataset('./data/SST/five/sst.fine.dev1.txt', w2id)
training_dataset = create_dataset('./data/SST/five/sst.fine.train1.txt', w2id)
test_dataset = create_dataset('./data/SST/five/sst.fine.test1.txt', w2id)
#  因为测试集、训练集、验证集形状、类型相同，这里根据数据集结构，构建了一个迭代器。
iterator = tf.data.Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)
sentence, labels = iterator.get_next()  # 经过convert_word2id的处理 sentence 是个元组，元组的第一个值是特征，第二个是有效单词数目
features = sentence[0]
sen_len = sentence[1]
# 可以使用不同的初始化操作，创建不同的数据来源
training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)
test_init_op = iterator.make_initializer(test_dataset)

with tf.name_scope('embedding'):
    embedding_matrix = tf.Variable(embedding, dtype=tf.float32, name='embedding', trainable=False)
    inputs = tf.nn.embedding_lookup(embedding_matrix, features)
    inputs = tf.nn.dropout(inputs, 0.8)

with tf.name_scope('lstm'):
    lstm_cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE)
    outputs, state = tf.nn.dynamic_rnn(
        cell=lstm_cell,
        inputs=inputs,
        sequence_length=sen_len,
        dtype=tf.float32
    )
    state = state.h  # shape:[batch_size, hidden_size]

with tf.name_scope('loss'):
    softmax_weights = tf.Variable(tf.random_uniform([HIDDEN_SIZE, nclass], -0.01, 0.01, dtype=tf.float32))
    softmax_bias = tf.Variable(tf.random_uniform([nclass], -0.01, 0.01, dtype=tf.float32))
    logits = tf.matmul(state, softmax_weights) + softmax_bias
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(loss) + L2_reg * tf.nn.l2_loss(softmax_weights)

with tf.name_scope('acc'):
    # probs = tf.nn.softmax(logits)  # 归一化前后，最大值位置不变
    new_labels = tf.cast(labels, tf.int64)
    accuracy = tf.equal(tf.argmax(logits, 1), new_labels)
    acc = tf.reduce_mean(tf.cast(accuracy, tf.float32))  # 需要注意的是，这里的样本数不一定等于BATCH_SIZE，因为文件尾，最后一个批次取不满。

with tf.name_scope('optimizer'):
    global_step = tf.Variable(0, name='tr_global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate).minimize(loss, global_step=global_step)

with tf.Session() as sess:
    # 在对话中写函数是一件危险的行为，注意不要在函数中操作图中的张量，否则造成图结构的冗余。
    sess.run(tf.global_variables_initializer())
    def validation():
        sess.run(validation_init_op)
        for i in range(1600):
            try:
                _, _loss, _global_step, _acc = sess.run([optimizer, loss, global_step, acc])
                print('after {} global_steps batch_avg_loss is {}, batch_avg_acc is {}'.format(_global_step, _loss, _acc))
            except tf.errors.OutOfRangeError:
                sess.run(validation_init_op)


    def test():
        sess.run(test_init_op)
        total_loss = 0
        total_acc = 0
        cnt = 0
        while True:
            try:
                _loss, _acc = sess.run([loss, acc])
                cnt += 1
                total_loss += _loss
                total_acc += _acc
                # print('batch_loss is : {} batch_acc is: {}'.format(_loss, _acc))
            except tf.errors.OutOfRangeError:
                print('total_loss is : {} acc is: {}'.format(total_loss/cnt, total_acc/cnt))  # 需要23个batch遍历一遍测试集
                break


    def train():
        sess.run(training_init_op)
        # write = tf.summary.FileWriter('./graphs', sess.graph)
        for i in range(max_iters):
            try:
                _, _loss, _global_step, _acc = sess.run([optimizer, loss, global_step, acc])
                #print('after {} global_steps batch_avg_loss is {}, batch_avg_acc is {}'.format(_global_step, _loss, _acc))
            except tf.errors.OutOfRangeError:
                test()  # 每遍历一遍训练集，进行一次测试
                sess.run(training_init_op)  # 初始化训练集迭代器，重新开始训练

    # validation()
    train()
