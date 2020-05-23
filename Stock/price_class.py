# -*- coding: utf-8 -*-
from __future__ import print_function
import datetime
import logging
logging.basicConfig(level=logging.DEBUG)
from hyperparams import Hyperparams as hp
from data_loader import get_stock_feature, price_batch
from price_modules import *
from utils import log
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
batch_size = 32
hidden_num = 8
log_write = open("{}/log_stock_price.txt".format(hp.log_stock_price_classify_path), "w")


class Graph(object):
    def __init__(self, dimension, is_training=True):
        tf.reset_default_graph()
        self.x = tf.placeholder(tf.float32, shape=(None, hp.Pmaxlen, dimension))
        self.y = tf.placeholder(tf.int32, shape=(None, 2))
        self.lr = tf.Variable(hp.lr, trainable=False)
        #self.learning_rate = tf.Variable(100.00, trainable=False)

        # Encoder
        with tf.variable_scope("encoder"):
            # Positional Encoding
            if hp.Psinusoid:
                self.input = self.x + positional_encoding(self.x,
                                                          num_units=dimension,
                                                          zero_pad=False,
                                                          scale=False,
                                                          scope="enc_pe")
            else:
                self.input = self.x + embedding(
                    tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                    vocab_size=hp.Pmaxlen,
                    num_units=dimension,
                    zero_pad=False,
                    scale=False,
                    scope="enc_pe")

            # tf.tile()的作用是改变输入的维度，第一个参数为输入，假设维度为[10,1],第二个参数为维度,[1,32],
            # input的维度就变为[10*1,1*32]

            # Dropout
            # self.input = tf.layers.dropout(self.input,
            #                                rate=hp.dropout_rate,
            #                                training=tf.convert_to_tensor(is_training))  # 按0.1概率选择

            # Blocks
            for i in range(hp.Pnum_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    # Multihead Attention
                    self.input = multihead_attention(queries=self.input,
                                                     keys=self.input,
                                                     num_units=hp.Phidden_units,
                                                     num_heads=hp.Pnum_heads,
                                                     dropout_rate=hp.Pdropout_rate,
                                                     is_training=is_training,
                                                     causality=False)

                    # Feed Forward
                    self.input = feedforward(self.input, num_units=[4 * hp.Phidden_units, hp.Phidden_units])
                    self.output = tf.reshape(self.input, [-1, hp.Pmaxlen * hp.Phidden_units])
        # Final linear projection
        self.logits = tf.layers.dense(self.output, 2)
        self.preds = tf.argmax(self.logits, dimension=-1)
        self.correct_predictions = tf.equal(self.preds, tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

        if is_training:
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                               labels=self.y))
            grad_clip = 5
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), grad_clip)
            train_op = tf.train.AdamOptimizer(self.lr)
            self.optimizer = train_op.apply_gradients(zip(grads, tvars))


if __name__ == '__main__':
    logger = log(hp.log_path, "1")
    train = True



    general = True



    comments_list = [
        '000333',
        '000423',
       '000651',
       '000858',
       # '000868',
        #'002607',
        #'300027',
        #'600519'
          ]
    dimen = 1
    """
    if general:
        if train:
            train_x_g = np.zeros((0, hp.Pmaxlen, dimen))
            train_y_g = np.zeros((0, 2))
            test_x_g = np.zeros((0, hp.Pmaxlen, dimen))
            test_y_g = np.zeros((0, 2))
            for stockID in comments_list:
                path = "{}/{}.txt".format(hp.feature_series_path, stockID)
                train_x, train_y, test_x, test_y, dimension = get_stock_feature(path,v=1,b=0,n=0,i=0)
                train_x_g = np.concatenate((train_x_g, train_x), axis=0)
                train_y_g = np.concatenate((train_y_g, train_y), axis=0)
                test_x_g = np.concatenate((test_x_g, test_x), axis=0)
                test_y_g = np.concatenate((test_y_g, test_y), axis=0)
                print('train_x', train_x.shape)
                print('train_y', train_y.shape)
            np.save('test_x_g.npy', test_x_g)
            np.save('test_y_g.npy', test_y_g)
        else:
            test_x_g = np.load('test_x_g.npy')
            test_y_g = np.load('test_y_g.npy')

    """

    for stockID in comments_list:
        test_x_name = 'test_x_' + stockID + '.npy'
        test_y_name = 'test_y_' + stockID + '.npy'
        path = "{}/{}.txt".format(hp.feature_series_path, stockID)
        if train:
            train_x, train_y, test_x, test_y, dimension = get_stock_feature(path,v=1,b=0,n=0,i=0)
            print('train_x', train_x.shape)
            print('train_y', train_y.shape)
            np.save(test_x_name, test_x)
            np.save(test_y_name, test_y)
        else:
            test_x = np.load(test_x_name)
            test_y = np.load(test_y_name)


        g = Graph(dimension=dimen, is_training=train)
        saver = tf.train.Saver()
        sess = tf.Session()

        if train:
            # 给训练出来的模型保存起来
            lr_list = []

            stock_price_model_path = hp.model3_path + stockID + "/"

            # 给训练过程中的日志保存起来
            log_path = stock_price_model_path + "log"
            if not os.path.exists(stock_price_model_path):

                os.makedirs(stock_price_model_path)
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            log_train = open(log_path + "/train.txt", "w")

            init = tf.global_variables_initializer()
            sess.run(init)
            step = 1

            for i in range(hp.Pnum_epochs):
                # 训练数据生成器
                batches_train = price_batch(train_x, train_y)


                # 随模型进行训练 降低学习率
                acc = []
                loss = []
                #sess.run(tf.assign(learning_rate, hp.lr * (0.95 ** i)))

                for batch_x, batch_y in batches_train:
                    lr = hp.Pfactor * (hp.Phidden_units ** (-0.5) * min(step ** (-0.5), step * hp.Pwarmup ** (-1.5)))

                    #lr_list.append(lr)
                    #sess.run(tf.assign(g.learning_rate, 0.00005*))

                    feed = {g.x: batch_x, g.y: batch_y, g.lr: lr}
                    batch_loss, batch_accuracy, _ = sess.run([g.loss, g.accuracy, g.optimizer], feed_dict=feed)
                    time_str = datetime.datetime.now().isoformat()
                    log_train.write("{}: epoch {} step {}, loss {:g}, acc {:g}\n".format(time_str, i, step, batch_loss, batch_accuracy))
                    #print("{}: epoch {} step {}, loss {:g}, acc {:g}".format(time_str, i, step, batch_loss, batch_accuracy))
                    step += 1
                    acc.append(batch_accuracy)
                    loss.append(batch_loss)
                acc_mean = np.mean(acc)
                loss_mean = np.mean(loss)
                print("epoch {} loss_mean {:g}, acc_mean {:g}\n".format(i, loss_mean, acc_mean))
                log_train.write("epoch {} loss_mean {:g}, acc_mean {:g}\n".format(i, loss_mean, acc_mean))
                #if acc_mean > 0.90:
                #    break
                saver.save(sess, stock_price_model_path, global_step=step)
                print(lr)

                batches_test = price_batch(test_x, test_y)
                acc = []
                loss = []
                for batch_x, batch_y in batches_test:
                    feed = {g.x: batch_x, g.y: batch_y}
                    batch_loss, batch_accuracy = sess.run([g.loss, g.accuracy], feed_dict=feed)
                    time_str = datetime.datetime.now().isoformat()
                    log_train.write("{}: step {}, acc {}".format(time_str, str(step), batch_accuracy))
                    #print("{}: step {}, acc {}".format(time_str, str(step), batch_accuracy))
                    step += 1
                    acc.append(batch_accuracy)
                    loss.append(batch_loss)

                acc_mean = np.mean(acc)
                loss_mean = np.mean(loss)
                print('test mean acc: ' + str(acc_mean)+'loss'+str(loss_mean))

            saver.save(sess, stock_price_model_path, global_step=step)
            sess.close()
            print("Done")

        else:
            stock_price_model_path = hp.model3_path + stockID + "/"
            log_path = stock_price_model_path + "log"
            log_train = open(log_path + "/train.txt", "a")

            print('开始测试...',stockID)
            log_train.write("\n开始测试\n")
            saver.restore(sess, tf.train.latest_checkpoint(stock_price_model_path))
            batches_test = price_batch(test_x, test_y)

            acc = []
            step = 0
            for batch_x, batch_y in batches_test:
                feed = {g.x: batch_x, g.y: batch_y}
                batch_accuracy = sess.run([g.accuracy], feed_dict=feed)
                time_str = datetime.datetime.now().isoformat()
                log_train.write("{}: step {}, acc {}".format(time_str, str(step), batch_accuracy))
                #print("{}: step {}, acc {}".format(time_str, str(step), batch_accuracy))
                step += 1
                acc.append(batch_accuracy)

            acc_mean = np.mean(acc)
            acc_max = np.mean(acc)
            print('mean acc: ' + str(acc_mean))
            log_train.write("{}: acc_mean {}, acc_max {}".format(datetime.datetime.now().isoformat(),
                                                                 acc_mean, acc_max))

            print('Done')