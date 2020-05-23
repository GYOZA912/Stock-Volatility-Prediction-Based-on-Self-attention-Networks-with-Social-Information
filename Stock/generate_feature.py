# -*- coding: utf-8 -*-
import os
import numpy as np
import copy
from math import isnan
from datetime import datetime
from hyperparams import Hyperparams as hp

STOCK_PRICE_DIR = "./data/101/price"  # 把股价放在这个文件夹下面
STOCK_FEATURE_SERIES = hp.new_feature_series_path  # 把处理好的股票的特征数据放到该文件夹下
STOCK_BAIDU_INDEX = './data/101/BaiduIndex'
STOCK_POS_ENG_SERIES = hp.new_pos_neg_comments_path


def read_stock_time_series(file_path):
    """
    stock_time_series list [("xxxx-xx-xx", x.xx),("xxxx-xx-xx", x.xx), ...] 按照时间从小到大
    :param file_path:
    :return:
    """
    stock_time_series = {}
    with open(file_path, 'r') as f:
        line = f.readline().strip()
        while line:
            new_time = line.split()[0]
            if len(line.split()[0].split('/'))>0:
                new_time = '-'.join(line.split()[0].split('/'))
                #print(new_time)
            stock_time_series[new_time] = float(line.split()[1])
            line = f.readline().strip()
    stock_time_series = sorted(stock_time_series.items(), key=lambda k: k[0])
    return stock_time_series


def generate_stock_price_radio_series(stock_id):
    # 读取按照时间从小到大排好序的股价信息s
    stock_time_series = read_stock_time_series(os.path.join(STOCK_PRICE_DIR, stock_id + '.txt'))


    stock_time = []
    stock_price = []
    for t, p in stock_time_series:
        #print(t,p)
        stock_time.append(str(t))
        stock_price.append(float(p))
    assert len(stock_time) == len(stock_price)

    # 生成股价的波动序列
    price_previous_day = stock_price[0]
    time_series = stock_time[1:]
    stock_price_radio_series = []
    for price in stock_price[1:]:
        volatility = (price - price_previous_day) / price_previous_day
        #print(volatility)
        volatility *= 5
        volatility += 0.0000001
        stock_price_radio_series.append(volatility)
        price_previous_day = price

    assert len(stock_price_radio_series) == len(time_series)
    assert len(stock_price) == len(time_series) + 1
    # time_series 不要第一天的数据
    # stock_price_radio_series 没有第一天的股价波动值
    # stock_price 全部的股价信息，包括第一天的，比上面两个list长度多1
    return (time_series, stock_price_radio_series, stock_price)


def generate_stock_feature_series(stock_id, time_series, norm_window_size):
    def normalize_zscore_lookback(series, window_size):
        series_normalized = copy.copy(series)
        series_normalized[0] = 0
        for i in range(1, len(series)):
            if i < window_size:
                series_normalized[i] = (series[i] - np.mean(series[0: (i + 1)])) / np.std(series[0: (i + 1)])
                if isnan(series_normalized[i]):
                    series_normalized[i] = 0
            else:
                series_normalized[i] = (series[i] - np.mean(series[(i - window_size + 1): (i + 1)])) / np.std(
                    series[(i - window_size + 1): (i + 1)])
                if isnan(series_normalized[i]):
                    series_normalized[i] = 0
        return series_normalized

    stock_id = stock_id + '.txt'
    filename = os.path.join(STOCK_POS_ENG_SERIES, stock_id)

    stock_emotion_radio_series = []
    stock_emotion_total_series = []
    baidu_search_series = []

    # 读取股票搜索指数信息
    baidu_index_path = os.path.join(STOCK_BAIDU_INDEX, stock_id)
    baidu_index_list = read_stock_time_series(baidu_index_path)

    baidu_index_time = []
    baidu_index = []
    for t, bi in baidu_index_list:
        #print(t,bi)
        baidu_index_time.append(str(t))
        baidu_index.append(float(bi))
    assert len(baidu_index_time) == len(baidu_index)
    #print('bi', baidu_index)
    print(time_series[0])#sj
    print(baidu_index_time[0])

    date_idx = 0
    for day, search_count in zip(baidu_index_time, baidu_index):
        try:
            if day < time_series[date_idx]:  # 有搜索但是没有股价，需要跳过 可能是周六周日
                continue
            elif day > time_series[date_idx]:  # 有股价但是没有股评数据，那就把搜索的指标补0
                while day > time_series[date_idx]:
                   # print('day',day)
                    #print(time_series[date_idx])
                    baidu_search_series.append(0)
                    date_idx += 1
        except IndexError:
            # Out of len（time_series） range, the loop ends
            break
        baidu_search_series.append(search_count)
        #print(baidu_search_series)
        date_idx += 1
    #print(baidu_search_series)
    date_idx = 0
    #print(baidu_search_series)
    with open(filename, 'r') as f:
        for line in f.readlines():
            date, none_cnt, pos_cnt, neg_cnt = line.split()
            none_cnt = int(none_cnt)
            pos_cnt = int(pos_cnt)
            neg_cnt = int(neg_cnt)
            date = datetime.strptime(date, '%Y-%m-%d').date()
            date = str(date)

            # Skip days market not open
            # time_series里面的时间是股价序列的时间
            # data 是股票评论的时间序列
            # 起始的那天是在time_series[0]和data/stock_pos_neg_series/股票代号.txt[0] 中取得最大值
            # 结束的那天就是
            try:
                if date < time_series[date_idx]:  # 有股评数据但是没有股价，需要跳过 可能是周六周日
                    continue
                elif date > time_series[date_idx]:  # 有股价但是没有股评数据，那就把股评的指标补0
                    while date > time_series[date_idx]:
                        stock_emotion_radio_series.append(0)
                        stock_emotion_total_series.append(0)
                        date_idx += 1
            except IndexError:
                # Out of len（time_series） range, the loop ends
                break

            # date = time_series[date_idx] 的情况
            bullishness = np.log(float(0.01 + pos_cnt) /float (0.01+ neg_cnt))
            stock_emotion_radio_series.append(bullishness)
            stock_emotion_total_series.append(pos_cnt + neg_cnt)
            date_idx += 1

    # Normalize bullishness and comment numbers to zscore
    # Window size always 5
    # 如果股价的开始天数 小于 股评的开始天数，此处可能就报错
    # assert len(stock_emotion_radio_series) == len(time_series)
    # assert len(stock_emotion_total_series) == len(time_series)
    # assert len(baidu_search_series) == len(time_series)

    stock_emotion_radio_series = normalize_zscore_lookback(stock_emotion_radio_series, norm_window_size)
    stock_emotion_total_series = normalize_zscore_lookback(stock_emotion_total_series, norm_window_size)
    baidu_search_series = normalize_zscore_lookback(baidu_search_series, norm_window_size)

    return (stock_emotion_radio_series, stock_emotion_total_series, baidu_search_series)


def generate_series(stock_id, norm_window_size):
    [time_series, stock_price_radio_series, stock_price] = generate_stock_price_radio_series(stock_id)

    [stock_emotion_radio_series,
     stock_emotion_total_series,
     baidu_search_series] = generate_stock_feature_series(stock_id, time_series, norm_window_size)

    return (time_series,
            stock_price_radio_series,
            stock_emotion_radio_series,
            baidu_search_series,
            stock_emotion_total_series,
            stock_price)


if __name__ == "__main__":
    """
    norm_window_size 表示生成序列数据时，参考前多少天的数据
    comments_list里面存放着需要生成的股票代号
    """
    norm_window_size = 16
    comments_list = ['000333',
                     '000423',
                     '000651',
                     '000858',
                     '000868',
                     '002607',
                     #'300027',
                     '600519'

                     ]

    for stock_id in comments_list:
        print('stock_id', stock_id)
        [time_series, stock_price_radio_series,
         stock_emotion_radio_series, baidu_search_series,
         stock_emotion_total_series, stock_price] = generate_series(stock_id, norm_window_size)
        print(len(time_series),len(stock_emotion_radio_series))
        print(stock_price_radio_series)


        fileName = os.path.join(STOCK_FEATURE_SERIES, stock_id + '.txt')
        with open(fileName, 'w') as f:
            f.write('time\t\tvolatility\t\tbullishness\t\tnumber\t\tindex\n')
            for (time, price, volatility, bullishness, number, index) in zip(time_series, stock_price, stock_price_radio_series,
                                                                      stock_emotion_radio_series,
                                                                      stock_emotion_total_series, baidu_search_series):
                # f.write(time.strftime("%Y-%m-%d"))
                f.write(time)
                f.write('\t\t')
                f.write(str(price))
                f.write('\t\t')
                f.write(str(volatility))
                f.write('\t\t')
                f.write(str(bullishness))
                f.write('\t\t')
                f.write(str(number))
                f.write('\t\t')
                f.write(str(index))
                f.write('\n')
        print("Generated stockID:{}".format(stock_id))
        print('stock_id', stock_id)
        print('-------')
