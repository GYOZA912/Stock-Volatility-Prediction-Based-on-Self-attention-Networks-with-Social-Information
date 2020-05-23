# -*- coding: utf-8 -*-
from __future__ import print_function
import datetime
import logging
logging.basicConfig(level=logging.DEBUG)
from hyperparams import Hyperparams as hp
from data_loader import new_get_stock_feature, price_batch
from price_modules import *
from utils import log
import os

batch_size = 32
hidden_num = 8
log_write = open("{}/log_stock_price.txt".format(hp.log_stock_price_classify_path), "w")


class Graph(object):
    def __init__(self, dimension, is_training=True):
        tf.reset_default_graph()
        self.x = tf.placeholder(tf.float32, shape=(None, hp.maxlen, dimension))
        self.y = tf.placeholder(tf.int32, shape=(None, 2))

        # Encoder
        with tf.variable_scope("encoder"):
            # Positional Encoding
            if hp.sinusoid:
                self.input = self.x + positional_encoding(self.x,
                                                          num_units=dimension,
                                                          zero_pad=False,
                                                          scale=False,
                                                          scope="enc_pe")
            else:
                self.input = self.x + embedding(
                    tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                    vocab_size=hp.maxlen,
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
            for i in range(hp.num_blocks_price):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    # Multihead Attention
                    self.input = multihead_attention(queries=self.input,
                                                     keys=self.input,
                                                     num_units=hp.hidden_units,
                                                     num_heads=hp.num_heads,
                                                     dropout_rate=hp.dropout_rate,
                                                     is_training=is_training,
                                                     causality=False)

                    # Feed Forward
                    self.input = feedforward(self.input, num_units=[4 * hp.hidden_units, hp.hidden_units])
                    self.output = tf.reshape(self.input, [-1, hp.maxlen * hp.hidden_units])
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
            train_op = tf.train.AdamOptimizer(hp.lr)
            self.optimizer = train_op.apply_gradients(zip(grads, tvars))


if __name__ == '__main__':
    logger = log(hp.log_path, "1")
    train = True
    general = True

    comments_list = [
                      #'000662',
                      #'002212',
                      #'002298',
                      #'300168',
                      #'600570',
                     # '600571',
                      #'600588',
                      #'600718',
                      #'601519',
                      '603881'
                     ]

    for stockID in comments_list:
        path = "{}/{}.txt".format(hp.feature_series_path, stockID)
        train_x, train_y, test_x, test_y, dimension = new_get_stock_feature(path )
        print('train_x', train_x.shape)
        print('train_y', train_y.shape)

        g = Graph(dimension=dimension, is_training=train)
        saver = tf.train.Saver()
        sess = tf.Session()

        if train:
            # 给训练出来的模型保存起来

            stock_price_model_path = hp.model3_path + 'general/'+ stockID + "/"

            # 给训练过程中的日志保存起来
            log_path = stock_price_model_path + "log"
            if not os.path.exists(stock_price_model_path):
                os.makedirs(stock_price_model_path)
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            log_train = open(log_path + "/train.txt", "w")

            # = tf.global_variables_initializer()
            #sess.run(init)
            general_path = hp.model3_path + 'general/'
            if general:
                saver.restore(sess, tf.train.latest_checkpoint(general_path))
            else:
                saver.restore(sess, tf.train.latest_checkpoint(stock_price_model_path))
            step = 0
            learning_rate = tf.Variable(hp.lr, trainable=False)
            log_train.write("开始训练 {}".format(stockID))
            print("开始训练 {}".format(stockID))
            for i in range(hp.price_num_epochs):
                # 训练数据生成器
                batches_train = price_batch(train_x, train_y)
                # 随模型进行训练 降低学习率
                sess.run(tf.assign(learning_rate, hp.lr * (0.95 ** i)))
                acc = []
                loss = []
                for batch_x, batch_y in batches_train:
                    feed = {g.x: batch_x, g.y: batch_y}
                    batch_loss, batch_accuracy, _ = sess.run([g.loss, g.accuracy, g.optimizer], feed_dict=feed)
                    time_str = datetime.datetime.now().isoformat()
                    log_train.write("{}: epoch {} step {}, loss {:g}, acc {:g}\n".format(time_str, i, step, batch_loss, batch_accuracy))
                    print("{}: epoch {} step {}, loss {:g}, acc {:g}".format(time_str, i, step, batch_loss, batch_accuracy))
                    step += 1
                    acc.append(batch_accuracy)
                    loss.append(batch_loss)
                acc_mean = np.mean(acc)
                loss_mean = np.mean(loss)
                print("epoch {} loss_mean {:g}, acc_mean {:g}\n".format(i, loss_mean, acc_mean))
                log_train.write("epoch {} loss_mean {:g}, acc_mean {:g}\n".format(i, loss_mean, acc_mean))
                if acc_mean > 0.90:
                    break
                if i % 2 == 0:
                    saver.save(sess, stock_price_model_path, global_step=step)



                #######eval
                batches_test = price_batch(test_x, test_y)
                acc_test = []
                print('##########################start_validate###################')

                for batch_x, batch_y in batches_test:
                    feed = {g.x: batch_x, g.y: batch_y}
                    batch_accuracy = sess.run([g.accuracy], feed_dict=feed)
                    time_str = datetime.datetime.now().isoformat()
                    log_train.write("{}: step {}, acc {}".format(time_str, str(step), batch_accuracy))
                    print("{}: step {}, acc {}".format(time_str, str(step), batch_accuracy))
                    step += 1
                    acc_test.append(batch_accuracy)

                acc_mean_test = np.mean(acc_test)
                acc_max_test = np.max(acc_test)
                print('mean acc: ' + str(acc_mean_test))
                log_train.write("{}: acc_mean {}, acc_max {}".format(datetime.datetime.now().isoformat(),
                                                                     acc_mean_test, acc_max_test))
                print('##########################end_validate###################')

            saver.save(sess, stock_price_model_path, global_step=step)
            sess.close()
            print("Done")

        else:
            stock_price_model_path = hp.model3_path + 'general/' +stockID + "/"
            log_path = stock_price_model_path + "log"
            log_train = open(log_path + "/train.txt", "a")

            print('开始测试...')
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
                print("{}: step {}, acc {}".format(time_str, str(step), batch_accuracy))
                step += 1
                acc.append(batch_accuracy)

            acc_mean = np.mean(acc)
            acc_max = np.mean(acc)
            print('mean acc: ' + str(acc_mean))
            log_train.write("{}: acc_mean {}, acc_max {}".format(datetime.datetime.now().isoformat(),
                                                                 acc_mean, acc_max))

            print('Done')