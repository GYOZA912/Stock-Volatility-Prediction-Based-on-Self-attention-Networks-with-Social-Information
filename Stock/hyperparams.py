# -*- coding: utf-8 -*-
import os


class Hyperparams:
    """
    Hyperparameters
    """
    # training
    Pbatch_size = 32
    Sbatch_size = 32  # alias = N
    lr = 0.00008  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory
    
    # model
    Smaxlen = 16
    Pmaxlen = 16

    Phidden_units = 64
    Shidden_units = 128# alias = C
    Smin_cnt = 10  # words whose occurred less than min_cnt are encoded as <UNK>.
    Pnum_blocks = 1
    Snum_blocks = 2

    Sfactor= 0.02
    Swarmup= 200
    Snum_epochs = 9
    Pfactor= 0.04
    Pwarmup= 200
    Pnum_epochs = 15


    Snum_heads = 2
    Pnum_heads = 1
    train_pro = 0.8
    Sdropout_rate = 0.5
    Pdropout_rate = 0.2
    Ssinusoid = False  # If True, use sinusoid. If false, positional embedding.
    Psinusoid = False
    price_position = 4

    # pathf#
    new_feature_series_path = "./data/101/feature_norm"
    new_pos_neg_comments_path = "./data/101/pos_neg"
    log_path = "./log/v1/sentiment_classify/"
    log_sentiment_classify_path = "./log/v1/sentiment_classify_result/"
    log_stock_price_classify_path = "./log/v1/stock_price_result/"
    feature_series_path = './data/101/feature_norm'
    vocab_path = "./data/v1/vocab/"
    stock_comments_path = 'data/stock_comments'
    emotion_comments_path = './data/101/comments/'
    pos_neg_comments_path = './data/v1/stock_pos_neg_series'

    model1_path = './model_sentiment/v1/emotion/'
    model2_path = './model_sentiment/v1/pos_neg_emotion/'
    model3_path = './model_price/v1/'
    path = [model1_path, model2_path, model3_path,
            vocab_path, log_path, log_sentiment_classify_path,
            emotion_comments_path, pos_neg_comments_path,
            feature_series_path, log_stock_price_classify_path]
    for p in path:
        if not os.path.exists(p):
            os.makedirs(p)