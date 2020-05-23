# -*- coding: utf-8 -*-
import os
import numpy as np
import copy
from math import isnan
from datetime import datetime
from hyperparams import Hyperparams as hp

STOCK_PRICE_DIR = "./data/stock_price"  # 把股价放在这个文件夹下面
STOCK_FEATURE_SERIES = hp.feature_series_path  # 把处理好的股票的特征数据放到该文件夹下
STOCK_BAIDU_INDEX = './data/stock_baidu_index'
STOCK_POS_ENG_SERIES = hp.pos_neg_comments_path


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
            stock_time_series[line.split()[0]] = float(line.split()[1])
            line = f.readline().strip()
    stock_time_series = sorted(stock_time_series.items(), key=lambda k: k[0])
    return stock_time_series


def generate_stock_price_radio_series(stock_id):
    # 读取按照时间从小到大排好序的股价信息
    stock_time_series = read_stock_time_series(os.path.join(STOCK_PRICE_DIR, stock_id + '.txt'))

    stock_time = []
    stock_price = []
    for t, p in stock_time_series:
        stock_time.append(str(t))
        stock_price.append(float(p))
    assert len(stock_time) == len(stock_price)

    # 生成股价的波动序列
    price_previous_day = stock_price[0]
    time_series = stock_time[1:]
    stock_price_radio_series = []
    for price in stock_price[1:]:
        volatility = (price - price_previous_day) / price_previous_day
        volatility *= 5
        volatility += 0.5
        stock_price_radio_series.append(volatility)
        price_previous_day = price

    assert len(stock_price_radio_series) == len(time_series)
    assert len(stock_price) == len(time_series) + 1
    # time_series 不要第一天的数据
    # stock_price_radio_series 没有第一天的股价波动值
    # stock_price 全部的股价信息，包括第一天的，比上面两个list长度多1
    return (stock_time, stock_price_radio_series, stock_price)
def get_derivative_series(time, series):
    #print (series)
    print('time', len(time))
    print('series', len(series))

    # 生成股价的波动序列
    series_previous_day = series[0]
    time = time[1:]
    de_series = []

    for each_day in series[1:]:
        #print(each_day)
        de_each_day = (each_day-series_previous_day+1)/(series_previous_day+1)
        #de_each_day *= 5           #????????//
        #de_each_day += 0.5
        de_series.append(de_each_day)
        series_previous_day = each_day

    assert len(de_series) == len(time)
    assert len(series) == len(time) + 1
    # time 不要第一天的数据
    # de_series 没有第一天的股价波动值
    # series 全部的股价信息，包括第一天的，比上面两个list长度多1
    return (time, de_series)



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
        baidu_index_time.append(str(t))
        baidu_index.append(float(bi))
    assert len(baidu_index_time) == len(baidu_index)

    date_idx = 0
    for day, search_count in zip(baidu_index_time, baidu_index):
        try:
            if day < time_series[date_idx]:  # 有搜索但是没有股价，需要跳过 可能是周六周日
                continue
            elif day > time_series[date_idx]:  # 有股价但是没有股评数据，那就把搜索的指标补0
                while day > time_series[date_idx]:
                    baidu_search_series.append(0)
                    date_idx += 1
        except IndexError:
            # Out of len（time_series） range, the loop ends
            break
        baidu_search_series.append(search_count)
        date_idx += 1

    date_idx = 0
    with open(filename, 'r') as f:
        for line in f.readlines():
            date, pos_cnt, neg_cnt = line.split()
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
            bullishness = np.log((float(1 + pos_cnt))/float((1 + neg_cnt)))
            stock_emotion_radio_series.append(bullishness)
            stock_emotion_total_series.append(pos_cnt + neg_cnt)
            date_idx += 1

    # Normalize bullishness and comment numbers to zscore
    # Window size always 5
    # 如果股价的开始天数 小于 股评的开始天数，此处可能就报错
    # assert len(stock_emotion_radio_series) == len(time_series)
    # assert len(stock_emotion_total_series) == len(time_series)
    # assert len(baidu_search_series) == len(time_series)
    print ('stock_emotion_ratio')
    _, stock_emotion_radio_series_de = get_derivative_series(time_series, stock_emotion_radio_series)
    print ('stock_total_ratio')
    _, stock_emotion_total_series_de = get_derivative_series(time_series, stock_emotion_total_series)
    print ('baidu_search_index')
    _, baidu_search_series_de = get_derivative_series(time_series, baidu_search_series)

    stock_emotion_radio_series = normalize_zscore_lookback(stock_emotion_radio_series, norm_window_size)
    stock_emotion_radio_series_de = normalize_zscore_lookback(stock_emotion_radio_series_de, norm_window_size)
    stock_emotion_total_series = normalize_zscore_lookback(stock_emotion_total_series, norm_window_size)
    stock_emotion_total_series_de = normalize_zscore_lookback(stock_emotion_total_series_de, norm_window_size)
    baidu_search_series = normalize_zscore_lookback(baidu_search_series, norm_window_size)
    baidu_search_series_de = normalize_zscore_lookback(baidu_search_series_de, norm_window_size)



    return (stock_emotion_radio_series,stock_emotion_radio_series_de, stock_emotion_total_series, stock_emotion_total_series_de, baidu_search_series,baidu_search_series_de)


def generate_series(stock_id, norm_window_size):
    [time_series, stock_price_radio_series, stock_price] = generate_stock_price_radio_series(stock_id)
    #time_series有两个！！！！！
    #TODO！！！！！！！

    [stock_emotion_radio_series,
     stock_emotion_radio_series_de,
     stock_emotion_total_series,
     stock_emotion_total_series_de,
     baidu_search_series,
     baidu_search_series_de] = generate_stock_feature_series(stock_id, time_series, norm_window_size)

    return (time_series,
            stock_price[1:],
            stock_price_radio_series,
            stock_emotion_radio_series[1:],
            stock_emotion_radio_series_de,
            baidu_search_series[1:],
            baidu_search_series_de,
            stock_emotion_total_series[1:],
            stock_emotion_total_series_de,
)


if __name__ == "__main__":
    """
    norm_window_size 表示生成序列数据时，参考前多少天的数据
    comments_list里面存放着需要生成的股票代号
    """
    norm_window_size = 10
    comments_list = ['000662',
                     '002212',
                     '002298',
                     '300168',
                     '600570',
                     #'600571',
                     '600588',
                     '600718',
                     '601519',
                     '603881'
                     ]

    for stock_id in comments_list:
        [time_series, p, p_de, b, b_de, i, i_de, s, s_de] = generate_series(stock_id, norm_window_size)
        print(len(time_series), len(p),len(p_de),len(b),len(b_de),len(i),len(i_de),len(s),len(s_de))

        fileName = os.path.join(STOCK_FEATURE_SERIES, stock_id + '.txt')
        with open(fileName, 'w') as f:
            f.write('time\t\tvolatility\t\tbullishness\t\tnumber\t\tindex\n')
            for (time_series, p, p_de, b, b_de, i, i_de, s, s_de) in zip(time_series, p, p_de, b, b_de, i, i_de, s, s_de):
                # f.write(time.strftime("%Y-%m-%d"))
                f.write(time_series)
                f.write('\t\t')
                f.write(str(p))
                f.write('\t\t')
                f.write(str(p_de))
                f.write('\t\t')
                f.write(str(b))
                f.write('\t\t')
                f.write(str(b_de))
                f.write('\t\t')
                f.write(str(i))
                f.write('\t\t')
                f.write(str(i_de))
                f.write('\t\t')
                f.write(str(s))
                f.write('\t\t')
                f.write(str(s_de))
                f.write('\n')
        print("Generated stockID:{}".format(stock_id))