# -*- coding: utf-8 -*-
"""
用于载入数据：
1.有情感与无情感数据
2.正向情感与负向情感数据
"""
import jieba
import xlrd
from xlrd import xldate_as_tuple
import random
import re
import os
import numpy as np
from hyperparams import Hyperparams as hp
from datetime import date, datetime
import json
import pickle
from collections import Counter

def get_stopwords():
    stop_words = set()
    with open(os.path.join(os.getcwd(), 'data/stopword.txt'), 'r') as f:
        line = f.readline()
        while line:
            stop_words.add(line.strip())
            line = f.readline()
    return stop_words


STOP_WORDS = get_stopwords()


def save_json(content, path):
    content = json.dumps(content)
    with open(path, "w") as f:
        f.write(content)
    print("save json: {}".format(path))


def load_json(path):
    with open(path, "r") as f:
        content = f.read()
        content = json.loads(content)
    return content


def save_pickle(content, path):
    with open(path, 'wb') as handle:
        pickle.dump(obj=content, file=handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, 'rb') as handle:
        content = pickle.load(handle)
    return content


def fenci(sentence):
    # 输入一个str 返回 去除停用词的 list
    string = '[A-Za-z0-9\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）【】「」：:；;《）《》“”()\]\[»〔〕-]+'
    sentence = re.sub(string, '', sentence)
    sentence = jieba.lcut(sentence)
    sentence = [w for w in sentence if w not in STOP_WORDS]
    return sentence


def load_data(data_path, class_model, train_prop=hp.train_pro):
    data = []
    words = []
    read_file = xlrd.open_workbook(data_path)
    table = read_file.sheet_by_index(0)
    titles = table.col_values(0)[0:]
    comments = table.col_values(1)[0:]
    sentiments = table.col_values(2)[0:]

    i = 0
    pos_count = 0
    neg_count = 0
    useless_count = 0
    while i < len(sentiments):
        # 把没有标签的跳过
        if sentiments[i] == '':
            i += 1
            continue
        # # 有comments的用comments，没有comments的用对应的titles
        # if comments[i] != '':
        #     comments_input = str(comments[i])
        # else:
        #     comments_input = str(titles[i])
        # # 如果len(comments_input) > 2
        # # 采用这种方式的话有5071条数据被跳过了， pos_count 4607 有情绪； neg_count 2647 无情绪
        # # 如果len(comments_input) > 1
        # # 采用这种方式的话有3883条数据被跳过了， pos_count 5476 有情绪； neg_count 2966 无情绪
        #
        # comments_input = fenci(comments_input)

        # 如果只用标题数据的话
        # 如果len(comments_input) > 1
        # 采用这种方式的话有889条数据被跳过了， pos_count 7811 有情绪； neg_count 3625 无情绪
        # 如果只用标题数据的话
        # 如果len(comments_input) > 2
        # 采用这种方式的话有2269条数据被跳过了， pos_count 6793 有情绪； neg_count 3263 无情绪
        #
        comments_input = str(titles[i])
        comments_input = fenci(comments_input)

        words.extend(comments_input)
        if len(comments_input) != 0:
            if class_model == 0:
                # 分类为有无情感
                if sentiments[i] == 0:
                    data.append((comments_input, [1, 0]))  # 无情感
                    neg_count += 1
                else:
                    data.append((comments_input, [0, 1]))  # 有情感
                    pos_count += 1
            if class_model == 1:
                # 分类为正负情感
                if sentiments[i] == 0:
                    i += 1
                    continue
                elif sentiments[i] == 1:
                    data.append((comments_input, [0, 1]))  # 正向情感
                    pos_count += 1
                else:
                    data.append((comments_input, [1, 0]))  # 负向情感
                    neg_count += 1
        else:
            useless_count += 1

        i += 1

    random.shuffle(data)
    train = np.array(data[:int(train_prop * len(data))])
    test = np.array(data[int(train_prop * len(data)):])

    comments_train = []
    labels_train = []
    for comment, label in train:
        comments_train.append(comment)
        labels_train.append(label)

    comments_test = []
    labels_test = []
    for comment, label in test:
        comments_test.append(comment)
        labels_test.append(label)

    return comments_train, labels_train, comments_test, labels_test, words


def get_vocab(words=None):
    word2int_path = "{}word2int.txt".format(hp.vocab_path)
    int2word_path = "{}int2word.txt".format(hp.vocab_path)

    if os.path.exists("{}word.pickle".format(hp.vocab_path)):
        # words = load_pickle("{}word.pickle".format(hp.vocab_path))
        word2int = load_json(word2int_path)
        int2word = load_json(int2word_path)
        # 把一部分特殊字符添加到停词表中了
        # with open('data/stopword.txt', "a") as f:
        #     for i in range(2, 104):
        #         f.write(int2word[str(i)] + "\n")
        return word2int, int2word
    else:
        comments_list = ['000662.xls',
                         '002212.xls',
                         '002298.xls',
                         '300168.xls',
                         '600570.xls',
                         '600571.xls',
                         '600588.xls',
                         '600718.xls',
                         '601519.xls',
                         '603881.xls',
                         ]

        for comment in comments_list:
            print("加入字典：{}".format(comment))
            data_path = os.path.join(hp.stock_comments_path, comment)
            readfile = xlrd.open_workbook(data_path)
            table = readfile.sheet_by_index(0)
            titles = table.col_values(1)[1:]
            comments = table.col_values(2)[1:]
            for t in titles:
                content = fenci(t)
                words.extend(content)
            for c in comments:
                content = fenci(c)
                words.extend(content)

        save_pickle(words, "{}word.pickle".format(hp.vocab_path))

        all_word_count = Counter(words)
        all_word_count = all_word_count.most_common()
        all_word = []
        for word, count in all_word_count:
            if count < 5:
                continue
            all_word.append(word)
        words = sorted(set(all_word))
        special_words = ['<PAD>', '<UNK>']
        vocab = special_words + words
        word2int = {w: i for i, w in enumerate(vocab)}
        int2word = {i: w for i, w in enumerate(vocab)}
        save_json(word2int, word2int_path)
        save_json(int2word, int2word_path)
        print("整个词典的大小为：{}".format(len(word2int)))
        return word2int, int2word


def to_full_batch(batch_data, word2int, sequence_length=hp.Smaxlen):

    # 对每个batch进行补全
    batch_size = len(batch_data)
    full_batch = np.full((batch_size, sequence_length), word2int['<PAD>'], np.int32)
    for row in range(batch_size):
        if len(batch_data[row]) > sequence_length:
            full_batch[row] = batch_data[row][:sequence_length]
        else:
            full_batch[row, :len(batch_data[row])] = batch_data[row]
    return full_batch


def get_batch(comments, labels, word2int, batch_size=hp.Sbatch_size):
    start = 0
    end = batch_size
    for _ in range(int(len(comments) / batch_size)):
        if end > len(labels):
            end = len(labels)
        batch_x = comments[start:end]
        batch_x_index = []
        for x in batch_x:
            x_index = []
            for word in x:
                if word not in word2int.keys():
                    continue
                else:
                    x_index.append(word2int[word])
            batch_x_index.append(x_index)
        batch_x_index = to_full_batch(batch_x_index, word2int)
        batch_y = labels[start:end]
        #print(batch_x_index.shape)
        #print (len(batch_y))
        yield batch_x_index, batch_y
        start += batch_size
        end += batch_size


def read_unlabeled_stock(data_path):
    readfile = xlrd.open_workbook(data_path)
    table = readfile.sheet_by_index(0)
    titles = table.col_values(0)[1:]
    comments = table.col_values(1)[1:]
    dates = table.col_values(2)[1:]
    dates_new = []
    comments_new = []
    titles_new = []
    for (t, c, d) in zip(titles, comments, dates):
        try:
            d_n = [datetime(*xldate_as_tuple(d, 0))]

        except TypeError:
            continue
        else:
            dates_new.append(d_n)
            comments_new.append(c)
            titles_new.append(t)

        #print(d_n)


    print(len(titles_new))
    print(len(comments_new))
    print(len(dates_new))
    preprocessed_data = []
    date_of_comments = []
    for (title, comment, day) in zip(titles_new, comments_new, dates_new):
        # Use title if comments are empty
        #print(type(comment))
        if str(comment).strip().strip(' 1234567890.e+') == "":
            comment = title
        # Skip if both title and comment are empty
        if str(comment).strip().strip('1234567890.e+-') == "" and str(title).strip().strip('1234567890\.e+') == "":
            continue

        #day = day.split()[0]  # 取到 xxxx-xx-xx
        #day = day.split(sep='-')
        day = date(int(day[0].year), int(day[0].month), int(day[0].day))
        #print(day)
        print(comment)

        try:
            comment = fenci(comment)
        except TypeError:
            continue
        else:
            preprocessed_data.append(comment)
            date_of_comments.append(day)

    assert len(preprocessed_data)==len(date_of_comments)


    return preprocessed_data, date_of_comments


def get_stock_feature(data_path, v=1, b=1, n=1, i=1):
    volatility = []
    bullishness = []
    number = []
    baidu_index = []
    with open(data_path, 'r') as f:
        line = f.readline()
        line = f.readline()
        while line:
            vol, bull, num, index = line.strip().split()[2:]
            volatility.append(float(vol))
            bullishness.append(float(bull))
            number.append(float(num))
            baidu_index.append(float(index))
            line = f.readline()

    preprocessed_data = []
    flag = []
    for m in range(2, int(len(baidu_index)/hp.Pmaxlen - 2)*hp.Pmaxlen):
        temp = []
        for j in range(m, m + hp.Pmaxlen):
            temp_chose = []
            if v:
                temp_chose.append(volatility[j])
            if b:
                temp_chose.append(bullishness[j])
            if n:
                temp_chose.append(number[j])
            if i:
                temp_chose.append(baidu_index[j])
            temp.append(temp_chose)
        preprocessed_data.append(temp)
        if float(volatility[m+hp.Pmaxlen]) > 0.0000001:
            flag.append([0, 1])
        else:
            flag.append([1, 0])

    # Separate data into train and test

    shuffle_indices = np.random.permutation(np.arange(int(hp.train_pro * len(preprocessed_data))))
    # preprocessed_data = np.array(preprocessed_data)
    train_x = np.array(preprocessed_data[:int(hp.train_pro * len(preprocessed_data))])[shuffle_indices]
    train_y = np.array(flag[:int(hp.train_pro * len(preprocessed_data))])[shuffle_indices]

    test_x = np.array(preprocessed_data[int(hp.train_pro * len(preprocessed_data)):])
    test_y = np.array(flag[int(hp.train_pro * len(preprocessed_data)):])

    dimension = v + b + n + i
    return train_x, train_y, test_x, test_y, dimension


def new_get_stock_feature(data_path, op=1, op_de =1, ob = 1, ob_de = 1, oi = 1, oi_de = 1, os = 1, os_de = 1):
    lp = []
    lp_de = []
    lb = []
    lb_de = []
    li = []
    li_de = []
    ls = []
    ls_de = []
    with open(data_path, 'r') as f:
        line = f.readline()
        line = f.readline()
        while line:
            p, p_de, b, b_de, i, i_de, s, s_de = line.strip().split()[1:]
            lp.append(float(p))
            lp_de.append(float(p_de))
            lb.append(float(b))
            lb_de.append(float(b_de))
            li.append(float(i))
            li_de.append(float(i_de))
            ls.append(float(s))
            ls_de.append(float(s_de))
            line = f.readline()

    preprocessed_data = []
    flag = []
    for m in range(2, int(len(li)/hp.maxlen - 2)*hp.maxlen):
        temp = []
        for j in range(m, m + hp.maxlen):
            temp_chose = []
            if op:
                temp_chose.append(lp[j])
            if op_de:
                temp_chose.append(lp_de[j])
            if ob:
                temp_chose.append(lb[j])
            if ob_de:
                temp_chose.append(lb_de[j])
            if oi:
                temp_chose.append(li[j])
            if oi_de:
                temp_chose.append(li_de[j])
            if os:
                temp_chose.append(ls[j])
            if os_de:
                temp_chose.append(ls_de[j])
            temp.append(temp_chose)
        preprocessed_data.append(temp)
        if float(lp_de[m+hp.maxlen]) > 0.5:
            flag.append([0, 1])
        else:
            flag.append([1, 0])
            # Separate data into train and test

    shuffle_indices = np.random.permutation(np.arange(int(hp.train_pro * len(preprocessed_data))))
    # preprocessed_data = np.array(preprocessed_data)
    train_x = np.array(preprocessed_data[:int(hp.train_pro * len(preprocessed_data))])[shuffle_indices]
    train_y = np.array(flag[:int(hp.train_pro * len(preprocessed_data))])[shuffle_indices]

    test_x = np.array(preprocessed_data[int(hp.train_pro * len(preprocessed_data)):])
    test_y = np.array(flag[int(hp.train_pro * len(preprocessed_data)):])

    dimension =  op + op_de + ob + ob_de + oi + oi_de + os + os_de
    return train_x, train_y, test_x, test_y, dimension

def price_batch(features, labels, batch_size=int(hp.Pbatch_size/2)):
    start = 0
    end = batch_size
    for _ in range(int(len(features) / batch_size)-1):
        batch_x = features[start:end]
        batch_y = labels[start:end]
        yield batch_x, batch_y
        start += batch_size
        end += batch_size


if __name__ == '__main__':
    # comments_train, labels_train, comments_test, labels_test, words = load_data(
    #                                                                   data_path='./data/labeled_data/comments.xlsx',
    #                                                                   class_model=1)
    # vocab, word2int, int2word = get_vocab(words)
    # train_data = get_batch(comments=comments_train,
    #                        labels=labels_train,
    #                        word2int=word2int)
    # for train in train_data:
    #     print(train)

    # a = 0
    # b = 0
    # for i in range(len(comments_train)):
    #     if labels_train[i] == [1, 0]:
    #         a += 1
    #     else:
    #         b += 1
    # print('正/负：{}/{}'.format(b, a))

    train_x, train_y, test_x, test_y, dimension = get_stock_feature('./stock_feature_series/000662.txt')
    print('Done!')
    print('ddd')