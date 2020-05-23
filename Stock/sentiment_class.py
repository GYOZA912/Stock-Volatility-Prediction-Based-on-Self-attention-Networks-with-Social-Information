# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import datetime
import sys
from hyperparams import Hyperparams as hp
from data_loader import load_data, get_vocab, get_batch, read_unlabeled_stock
from modules import *
from utils import log
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
log_sentiment_classify = open("{}train.txt".format(hp.log_sentiment_classify_path), "a")

#######
class Graph(object):
    def __init__(self, word2int, int2word, is_training=True):
        self.word2int = word2int
        self.int2word = int2word

        self.x = tf.placeholder(tf.int32, shape=(None, hp.Smaxlen))
        self.y = tf.placeholder(tf.int32, shape=(None, 2))
        self.learning_rate = tf.Variable(100.00, trainable=False)
        # Encoder
        with tf.variable_scope("encoder"):
            # Embeddingcdcd
            self.input = embedding(self.x,
                                   vocab_size=len(self.word2int),
                                   num_units=hp.Shidden_units,
                                   scale=True,
                                   scope="enc_embed")

            # Positional Encoding
            if hp.Ssinusoid:
                self.input += positional_encoding(self.x,
                                                  num_units=hp.Shidden_units,
                                                  zero_pad=False,
                                                  scale=False,
                                                  scope="enc_pe")
            else:
                self.input += embedding(
                    tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                    vocab_size=hp.Smaxlen,
                    num_units=hp.Shidden_units,
                    zero_pad=False,
                    scale=False,
                    scope="enc_pe")

            # Dropout
            self.input = tf.layers.dropout(self.input,
                                           rate=hp.Sdropout_rate,
                                           training=tf.convert_to_tensor(is_training))  # 按0.1概率选择

            # Blocks
            for i in range(hp.Snum_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    # Multihead Attention
                    self.input = multihead_attention(queries=self.input,
                                                     keys=self.input,
                                                     num_units=hp.Shidden_units,
                                                     num_heads=hp.Snum_heads,
                                                     dropout_rate=hp.Sdropout_rate,
                                                     is_training=is_training,
                                                     causality=False)

                    # Feed Forward
                    self.input = feedforward(self.input, num_units=[4 * hp.Shidden_units, hp.Shidden_units])
                    self.output = tf.reshape(self.input, [-1, hp.Smaxlen * hp.Shidden_units])
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
            train_op = tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer = train_op.apply_gradients(zip(grads, tvars))


if __name__ == '__main__':
    logger = log(hp.log_path, "1")
    """
       重要参数:
       class_model:
           0: 训练的是有无情感的分类模型
           1: 训练的是正负情感的分类模型
       train:
           0:训练
           1:测试
           2:利用模型去预测
        ps：class_model = 1 train = 2 不可选，此功能运行 generate_posneg.py
       """
    class_model = 1 # 可选参数： 0 1
    train = 1
    # 可选参数：0 1 2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    if class_model == 0:
        model_path = hp.model1_path
    else:
        model_path = hp.model2_path

    #
    if train == 0:

        comments_train, labels_train, comments_test, labels_test, words = load_data(
            data_path='./data/labeled_data/comments.xlsx',
            class_model=class_model)
        np.save("comments_test.npy", comments_test)
        np.save("labels_test.npy", labels_test)
        np.save("words.npy", words)
        word2int, int2word = get_vocab(words)
        g = Graph(word2int, int2word, is_training=True)
    else:

        comments_test = np.load("comments_test.npy")
        labels_test = np.load("labels_test.npy")
        words = np.load("words.npy")
        word2int, int2word = get_vocab(words)
        g = Graph(word2int, int2word, is_training=False)

    saver = tf.train.Saver()
    sess = tf.Session()
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    # 开始训练
    if train == 0:
        lr_list = []
        logger.info("开始训练 ... ")
        sess.run(tf.global_variables_initializer())
        step = 1
        for i in range(hp.Snum_epochs):
            # 训练数据生成器
            batches_train = get_batch(comments_train, labels_train, word2int)
            # 随模型进行训练 降低学习率



            acc = []
            loss = []
            for batch_x, batch_y in batches_train:
                lr = hp.Sfactor * (hp.Shidden_units ** (-0.5) * min(step ** (-0.5), step * hp.Swarmup ** (-1.5)))
                lr_list.append(lr)
                sess.run(tf.assign(g.learning_rate, lr))
                print('lr', lr)
                feed = {g.x: batch_x, g.y: batch_y}
                batch_loss, batch_accuracy, _ = sess.run([g.loss, g.accuracy, g.optimizer], feed_dict=feed)
                logger.info("{}: epoch {}, step {}, loss {:g}, acc {:g}".format(datetime.datetime.now().isoformat(),i, step, batch_loss, batch_accuracy))
                step += 1
                acc.append(batch_accuracy)
                loss.append(batch_loss)
            acc_mean = np.mean(acc)
            acc_max = np.max(acc)
            loss_mean = np.mean(loss)
            # 把结果写进文件中
            log_sentiment_classify.write("epoch {},acc_mean {:g},acc_max {:g},loss_mean {:g} \n".format(
                 i, acc_mean, acc_max, loss_mean))

            logger.info("开始测试 ... ")

            batches_test = get_batch(comments_test, labels_test, word2int)

            flags = []
            acc = []
            for batch_x, batch_y in batches_test:
                feed = {g.x: batch_x, g.y: batch_y}
                pre, batch_accuracy = sess.run([g.preds, g.accuracy], feed_dict=feed)
                logger.info("{}: step {}, acc {:g}".format(datetime.datetime.now().isoformat(), step, batch_accuracy))
                step += 1
                flags.extend(pre)
                acc.append(batch_accuracy)

            acc_mean = np.mean(acc)
            acc_max = np.max(acc)
            # 把结果写进文件中
            logger.info("{}: acc_mean {:g}, acc_max {:g}\n".format(
                i, acc_mean, acc_max))

            if acc_mean > 0.95:
                break
            saver.save(sess, model_path, global_step=step)
        saver.save(sess, model_path, global_step=step)
        sess.close()
        logger.info("Done")

    elif train == 1:
        logger.info("开始测试 ... ")
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        batches_test = get_batch(comments_test, labels_test, word2int)


        flags = []
        acc = []
        step = 0
        for batch_x, batch_y in batches_test:
            feed = {g.x: batch_x, g.y: batch_y}
            pre, batch_accuracy = sess.run([g.preds, g.accuracy], feed_dict=feed)
            logger.info("step {}, acc {:g}".format(step, batch_accuracy))
            step += 1
            flags.extend(pre)
            acc.append(batch_accuracy)

        acc_mean = np.mean(acc)
        acc_max = np.max(acc)
        for j in range(32):
            print('PRE', comments_test[j])
            print('LABEL', labels_test[j][1])
            print('FLAG', flags[j])
        # 把结果写进文件中
        logger.info("{}: acc_mean {:g}, acc_max {:g}\n".format(
            datetime.datetime.now().isoformat(), acc_mean, acc_max))
        """
        pos = 0
        neg = 0
        for label in range(len(labels_test)):
            if labels_test[label] == [1, 0]:
                neg += 1
            else:
                pos += 1
        logger.info("Done")
        """

    else:
        logger.info("有无情绪模型 开始预测 ... ")
        log_sentiment_classify.write("有无情绪模型 开始预测 ... \n")
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        comments_list = ['000662.xls',
                         #'002212.xls',
                         #'002298.xls',
                         #'300168.xls',
                         #'600570.xls',
                         #'600571.xls',
                         #'600588.xls',
                         #'600718.xls',
                         #'601519.xls',
                         #'603881.xls',
                         ]
        results_list = ['000662.txt',
                        #'002212.txt',
                        #'002298.txt',
                        #'300168.txt',
                        #'600570.txt',
                        #'600571.txt',
                        #'600588.txt',
                        #'600718.txt',
                        #'601519.txt',
                        #'603881.txt',
                        ]
        kk = 1
        for (comment, result) in zip(comments_list, results_list):
            logger.info("{} 当前股票: {} ... ".format(kk, result))
            preprocessed_data, date_of_comments = read_unlabeled_stock(os.path.join(hp.stock_comments_path, comment))
            total = int(len(preprocessed_data) / hp.Sbatch_size)
            preprocessed_data = preprocessed_data[:total * hp.Sbatch_size]
            date_of_comments = date_of_comments[:total * hp.Sbatch_size]

            preprocessed_data_test = preprocessed_data[:32]
            date_of_comments = date_of_comments[:32]
            flags = []
            #batches_unlabeled = get_batch(preprocessed_data, preprocessed_data, word2int)   # 第二个参数是不用的
            batches_unlabeled = get_batch(preprocessed_data_test, preprocessed_data_test, word2int)  # 第二个参数是不用的
            j = 1
            for batch_x, _ in batches_unlabeled:
                #if j % 30 == 0:
                #    print('正在预测：{}/{}'.format(j, total))
                feed = {g.x: batch_x}
                pre_flag = sess.run(g.preds, feed_dict=feed)
                flags.extend(pre_flag)
                j += 1
            print('len(flag)', len(flags))
            for i in range(len(preprocessed_data_test)):
                print('pre', preprocessed_data_test[i])
                print('flag', flags[i])
            """
            pos_comments_num = 0
            with open(os.path.join(hp.emotion_comments_path, result), 'w', encoding='utf-8') as f:
                print('写入带有情绪的评论')
                for num_ in range(len(flags)):
                    if flags[num_] >= 0.5 and len(preprocessed_data[num_]) >= 1:
                        # 只写入分类成有情绪的评论和对应的日期
                        pos_comments_num += 1
                        f.write(str(date_of_comments[num_]) + '\t')
                        f.write(' '.join(preprocessed_data[num_]) + '\n')

            logger.info("{}: stock {}, 正样本个数/总个数: {}/{} {:g}".format(
                datetime.datetime.now().isoformat(),
                result, pos_comments_num, len(flags), float(pos_comments_num)/len(flags)))

            log_sentiment_classify.write("\n{}: stock {}, 正样本个数/总个数: {}/{} {:g}% \n".format(
                datetime.datetime.now().isoformat(),
                result, pos_comments_num, len(flags), float(pos_comments_num)/len(flags)))

            # TODO: Display checked comments
            print("Successfully generated sentiment series for %s" % comment)
            kk += 1
            """