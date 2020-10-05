import pickle

from cz.data import *
from cz.progressbar import ProgressBar
from libsvm.svmutil import *
import numpy as np
import configparser
import datetime
from cz.string import *

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def get_label(kernel=2,C=1,gamma=0.001):
    kernel_list = ['linear','polynomial','rbf']
    if kernel >= 1:
        return '_%s_C_%s_gamma_%s'%(kernel_list[kernel],str(C),str(gamma))
    else :
        return '_%s_C_%s' % (kernel_list[kernel],str(C))


def get_par(kernel=2,C=1,gamma=0.001):
    print('>>>>>>>>>>>>>>>>>>> start: '+get_label(kernel,C,gamma)+' at: '+datetime.datetime.now().strftime('%T'))

    if kernel >= 1:
        return '-t %i -c %f -g %f -v 5 -h 0' % (kernel,C,gamma)
    else:
        return '-t %i -c %f -v 5 -h 0' % (kernel,C)

def load_data(n_frame: int = 1):

    print('>>>>>>> loading data')
    cf = configparser.ConfigParser()
    cf.read('config.ini')

    frame_postfix = '_' + str(n_frame) + 'f'

    x_train = cf.get('dataset', 'x_train' + frame_postfix)
    y_train = cf.get('dataset', 'y_train' + frame_postfix)
    x_valid = cf.get('dataset', 'x_valid' + frame_postfix)
    y_valid = cf.get('dataset', 'y_valid' + frame_postfix)
    x_test = cf.get('dataset', 'x_test' + frame_postfix)
    y_test = cf.get('dataset', 'y_test' + frame_postfix)

    rval = [x_train, y_train, x_test, y_test]

    for i in range(len(rval)):
        with open(rval[i], 'rb') as f:
            rval[i] = pickle.load(f)

    print('<<<<<<< load data done')

    return rval


if __name__ == '__main__':

    x_train, y_train, x_test, y_test = load_data(2)

    scale = np.max(x_train)
    x_train /= scale
    x_test /= scale

    print('>>>>>>> creating svm problem')
    prob = svm_problem(y_train, x_train)
    print('<<<<<<< done')
    print('>>>>>>> start training')
    param = svm_parameter('-c 32 -g 0.01 -b 1')
    model = svm_train(prob, param)
    svm_save_model('model/svm_hog', model)
    print('<<<<<<< end training')
    # model = svm_load_model('model/svm')
    print('>>>>>>> start predicting')
    pred_labels, pred_acc, pred_values = svm_predict(y_test, x_test, model, '-b 1')
    print('<<<<<<< end predicting')

    pred_values = np.array(pred_values)[:, 1].ravel()
    plots = get_roc_points(pred_values, y_test)
    with open('points/SVM_c_32_g_0.01_hog_80000.pkl', 'wb') as f:
        pickle.dump((plots[:, 0], plots[:, 1]), f)
    # with open('points/SVM_c_32_g_0.01_hog.pkl', 'rb') as f:
    #     plots_x, plots_y = pickle.load(f)
    # plt.plot(plots_x, plots_y, label='c_32_g_0.01')
    plt.plot(plots[:, 0], plots[:, 1], label='c_32_g_0.01')
    plt.xscale('log')
    plt.xlabel('false alarm')
    plt.ylabel('missing rate')
    plt.title('ROC curve')
    plt.grid()
    plt.legend()
    plt.show()

    # for kernel in [0, 1, 2]:
    #     for C in [0.25, 1, 4, 16]:
    #         if kernel >= 1:
    #             for gamma in [1,0.1,0.001,0.0001]:
    #                 counter += 1
    #                 if counter % 3 != my_code or counter < 28:
    #                     continue
    #                 param = svm_parameter(get_par(kernel,C,gamma))
    #                 acc = svm_train(prob, param)
    #                 print('<<<<<<<<<<<<<<<<<<<<<  acc=', acc,'%')
    #                 with open('SVM_CV/svm_validation'+get_label(kernel,C,gamma)+'.pkl','wb') as f:
    #                     pickle.dump(acc,f)
    #         else:
    #             counter += 1
    #             if counter % 3 != my_code or counter < 28:
    #                 continue
    #             param = svm_parameter(get_par(kernel, C))
    #             acc = svm_train(prob, param)
    #             print('<<<<<<<<<<<<<<<<<<<<< acc=', acc,'%')
    #             with open('SVM_CV/svm_validation' + get_label(kernel, C) + '.pkl','wb') as f:
    #                 pickle.dump(acc, f)

