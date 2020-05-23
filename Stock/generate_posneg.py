# -*- coding: utf-8 -*-
from hyperparams import Hyperparams as hp
from sentiment_class import Graph
from data_loader import get_vocab, get_batch, read_unlabeled_stock
from modules import *
import os
import datetime


log_sentiment_classify = open("{}train.txt".format(hp.log_sentiment_classify_path), "a")


def main():

    word2int, int2word = get_vocab()
    print("vocabulary loaded")

    g = Graph(word2int, int2word, is_training=True)
    print("Graph loaded")

    saver1 = tf.train.Saver()
    sess1 = tf.Session()
    saver1.restore(sess1, tf.train.latest_checkpoint(hp.model1_path))

    saver2 = tf.train.Saver()
    sess2 = tf.Session()
    saver2.restore(sess2, tf.train.latest_checkpoint(hp.model2_path))

    comments_list = [#'000333.xlsx',
                     #'000423.xlsx',
                     #'000651.xlsx',
                     #'000858.xlsx',
                     #'000868.xlsx',
                     #'002607.xlsx',
                     #'002739.xlsx', File is not a zip file
                     '300027.xlsx',
                     #'600518.xlsx', No such file or directory: './data/101/comments/600518.xlsx'
                     '600519.xlsx'

                     ]
    ##
    results_list = [#'000333.txt',
                    #'000423.txt',
                    #'000651.txt',
                    #'000858.txt',
                    #'000868.txt',
                    #'002607.txt',
                    #'002739.txt',
                    '300027.txt',
                    #'600518.txt',
                    '600519.txt'

                    ]

    """
    comments_list = ['000662.txt',
                     '002212.txt',
                     '002298.txt',
                     '300168.txt',
                     '600570.txt',
                     '600571.txt',
                     '600588.txt',
                     '600718.txt',
                     '601519.txt',
                     '603881.txt',
                     ]
    results_list = ['000662.txt',
                    '002212.txt',
                    '002298.txt',
                    '300168.txt',
                    '600570.txt',
                    '600571.txt',
                    '600588.txt',
                    '600718.txt',
                    '601519.txt',
                    '603881.txt',
                    ]
    """
    kk = 1
    for (comment_dir, result_dir) in zip(comments_list, results_list):
        print(kk, comment_dir)

        comment_name = hp.emotion_comments_path + comment_dir
        preprocessed_data, date_of_comments = read_unlabeled_stock(comment_name)
        print(preprocessed_data[3])
        print(date_of_comments[3])

        total = int(len(preprocessed_data) / hp.Sbatch_size)
        preprocessed_data = preprocessed_data[:total * hp.Sbatch_size]
        date_of_comments = date_of_comments[:total * hp.Sbatch_size]

        flags1 = []
        flags2 = []
        batches_unlabeled = get_batch(preprocessed_data, preprocessed_data, word2int)
        j = 1
        for batch_x, _ in batches_unlabeled:
            if j % 300 == 0:
                print('正在预测：{}/{}'.format(j, total))

            pre_flag1 = sess1.run(g.preds, feed_dict={g.x: batch_x})
            pre_flag2 = sess2.run(g.preds, feed_dict={g.x: batch_x})
            flags1.extend(pre_flag1)
            flags2.extend(pre_flag2)
            j += 1
        pos_num = 0
        neg_num = 0
        none_num = 0
        date2sentimentCount = dict()
        for (flag1, flag2, day) in zip(flags1, flags2, date_of_comments):
            if day not in date2sentimentCount:
                date2sentimentCount[day] = [0, 0, 0]  # NONE, POS, NEG
            if flag1 == 0:
                date2sentimentCount[day][0] += 1  # num_NONE + 1
                none_num += 1
            else:
                if flag2 == 1:
                    date2sentimentCount[day][1] += 1 #num_POS + 1
                    pos_num += 1
                else:
                    date2sentimentCount[day][2] += 1  # num_NEG + 1
                    neg_num += 1

        log_sentiment_classify.write("{}: stock:{} pos {}, neg {}, total {}\n".format(
            datetime.datetime.now().isoformat(), result_dir, pos_num, neg_num, len(flags1)))

        print('开始写入啦！')
        with open(os.path.join(hp.new_pos_neg_comments_path, result_dir), 'w') as f:
            s = sorted(date2sentimentCount.items(), key=lambda x: x[0])

            for (day, num) in s:
                #print(day, num)
                f.write(str(day))
                f.write('\t')
                f.write(str(num[0]))  # NONE
                f.write('\t')
                f.write(str(num[1]))  # 正
                f.write('\t')
                f.write(str(num[2]))  # 负
                f.write('\n')

        # TODO: Display checked comments
        #logger.info("pos neg for %s" % comment_dir)
        kk += 1


if __name__ == '__main__':
    main()
