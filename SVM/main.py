import configparser
import pickle
from configparser import ConfigParser

from libsvm.svmutil import *

from cz.progressbar import *

import numpy as np

def load_data(dataset, n_frame: int = 1):

    print('>>>>>>> loading data')
    cf = ConfigParser()
    cf.read(dataset)

    frame_postfix = '_' + str(n_frame) + 'f'

    x_train = cf.get('dataset', 'x_train' + frame_postfix)
    y_train = cf.get('dataset', 'y_train' + frame_postfix)
    x_test = cf.get('dataset', 'x_test' + frame_postfix)
    y_test = cf.get('dataset', 'y_test' + frame_postfix)

    rval = [x_train, y_train, x_test, y_test]

    for i in range(len(rval)):
        with open(rval[i], 'rb') as f:
            rval[i] = pickle.load(f)

    print('<<<<<<< load data done')

    return rval


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data('config.ini', n_frame=2)

    x_train = x_train[:40000]
    y_train = y_train[:40000]

    dataset_size = x_train.shape[0]

    # scale = np.max(x_train)
    # x_train /= scale
    row = ''
    pb = ProgressBar(dataset_size)
    pb.start()
    for i in range(dataset_size):
        row += str(y_train[i])
        for j in range(x_train.shape[1]):
            row += ' ' + str(j + 1) + ':' + str(x_train[i, j])
        row += '\n'
        pb.show_progress(i + 1)
    pb.stop()

    f = open('dataset_40000', 'w')
    f.write(row)



