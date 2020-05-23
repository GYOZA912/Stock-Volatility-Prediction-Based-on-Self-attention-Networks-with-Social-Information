# -*- coding: utf-8 -*-

from __future__ import print_function
import datetime
import logging
logging.basicConfig(level=logging.DEBUG)
from hyperparams import Hyperparams as hp

from price_modules import *
from sklearn import svm
from sklearn.linear_model.logistic import LogisticRegression
from data_loader import new_get_stock_feature, price_batch
import os

batch_size = 32
hidden_num = 8
log_write = open("{}/log_stock_price.txt".format(hp.log_stock_price_classify_path), "w")




if __name__ == '__main__':

    train = False
    comments_list = [
                     #'000662',
                     #'002212',
                     #'002298',
                     #'300168',
                     #'600570',
                     #'600571',
                     #'600588',
                     #'600718',
                     #'601519',
                     '603881'
                     ]
    for stockID in comments_list:
        path = "{}/{}.txt".format(hp.feature_series_path, stockID)
        train_x, train_y, test_x, test_y, dimension = new_get_stock_feature(path)
        print('train_x',train_x.shape)
        print('train_y',train_y.shape)
        print(stockID)
        train_x = np.reshape(train_x,(train_x.shape[0],-1))
        test_x = np.reshape(test_x, (test_x.shape[0], -1))
        train_y = train_y[:,0]
        test_y = test_y[:,0]
        clf = svm.SVC(gamma='auto')
        clf.fit(train_x, train_y)

        SVM_pred = clf.predict(test_x)
        SVM_acc = np.mean(np.equal(SVM_pred, test_y, dtype='int') + 0)
        print(stockID, 'SVM_acc', SVM_acc)

        classifier = LogisticRegression()
        classifier.fit(train_x, train_y)
        Log_pred = classifier.predict(test_x)
        Log_acc = np.mean(np.equal(Log_pred, test_y, dtype='int') + 0)
        #print(stockID, 'Log_acc', Log_acc)