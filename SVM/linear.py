import configparser
import pickle
from operator import itemgetter

from liblinear.liblinearutil import *
from cz.string import *
from cz.data import *
from cz.progressbar import *

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np


def load_data(dataset):

    print('>>>>>>> loading data')
    cf = configparser.ConfigParser()
    cf.read(dataset)


    x_train = cf.get('dataset', 'x_train_2f')
    y_train = cf.get('dataset', 'y_train_2f')
    x_test = cf.get('dataset', 'x_test_2f')
    y_test = cf.get('dataset', 'y_test_2f')

    rval = [x_train, y_train, x_test, y_test]

    for i in range(len(rval)):
        with open(rval[i], 'rb') as f:
            rval[i] = pickle.load(f)

    print('<<<<<<< load data done')

    return rval

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data('config.ini')

    x_train = x_train[:40000]
    y_train = y_train[:40000]

    scale_factor = np.max(x_train)
    x_train /= scale_factor
    x_test /= scale_factor
    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1

    x = [0.25, 0.8, 0.9, 1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2, 2.2, 2.4, 2.5, 2.6, 2.7, 2.8, 3, 4]
    # x = [2]
    acc = []
    for c in x:
        path = generate_filename(dir='linear', basename='cv_accuracy', tag={'c': c}, postfix='pkl')
        with open(path, 'rb') as f:
            acc.append(pickle.load(f))
    plt.plot(x, acc, 'o-')
        # print('>>>>>>> start c = ' + str(c))
        # model = train(y_train, x_train, '-c ' + str(c))
        # print('<<<<<<< end c = ' + str(c), str(model))
        # save_model(generate_filename(dir='linear', basename='model', tag={'c': c}, overwrite=True), model)
        # model = load_model(generate_filename(dir='linear', basename='model', tag={'c': c}))
        #
        # print('>>>>>>> start predict')
        # pred_labels, pred_acc, pred_values = predict(y_test, x_test, model)
        # print('<<<<<<< predict done')
        # labels = model.get_labels()
        # pred_values = [labels[0] * val[0] for val in pred_values]
        # pred_values = np.array(pred_values).ravel()
        # print(pred_values.shape)
        # pred_values.sort()
        # print(pred_values)

        # db = []
        # pos = neg = 0
        # for i in range(len(y_test)):
        #     if y_test[i] > 0:
        #         pos += 1
        #     else:
        #         neg += 1
        #     db.append([pred_values[i], y_test[i]])
        #
        # # sorting by decision value
        # db = sorted(db, key=itemgetter(0), reverse=True)
        #
        # # calculate ROC
        # xy_arr = np.zeros((len(db), 2))
        # tp, fp = 0., 0.  # assure float division
        # for i in range(len(db)):
        #     if db[i][1] > 0:  # positive
        #         tp += 1
        #     else:
        #         fp += 1
        #     xy_arr[i] = fp / neg, 1 - tp / pos
        # with open(generate_filename(dir='plots', basename='liblinear', tag={'c': c}, overwrite=True), 'wb') as f:
        #     var = (xy_arr[:, 0], xy_arr[:, 1])
        #     pickle.dump(var, f)
        # plt.plot(xy_arr[:, 0], xy_arr[:, 1])

    #     thresholds = np.unique(pred_values)
    #     plots = np.zeros((thresholds.shape[0], 2))
    #     pb = ProgressBar(thresholds.shape[0])
    #     pb.start()
    #     for i in range(thresholds.shape[0]):
    #         TP, TN, FP, FN = analyze_prob(pred_values, y_test, thresholds[i])
    #         missing_rate = FN / (TP + FN)
    #         false_alarm = FP / (TN + FP)
    #         plots[i, 0] = false_alarm
    #         plots[i, 1] = missing_rate
    #         pb.show_progress(i + 1)
    #     pb.stop()
    #     plt.plot(plots[:, 0], plots[:, 1], label='c_' + str(c))
    # plt.xscale('log')
    plt.grid()
    plt.show()
        # acc.append(model)
        # save_pickle(model, dir='linear', basename='model', tag={'c': c}, overwrite=True)
        # acc.append(load_pickle(generate_filename(dir='linear', basename='cv_accuracy', tag={'c': c}, postfix='pkl')))
        # print(c, acc[len(acc) - 1])
    # plt.plot(x, acc, '.')
    # plt.grid()
    # plt.show()
