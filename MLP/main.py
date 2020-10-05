import gzip
import random
import cv2
from configparser import ConfigParser

from mlp import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from cz.data import *

def load_mnist_data(dataset):
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    def shared_dataset(data_xy):
        data_x, data_y = data_xy
        x = np.asarray(data_x, dtype=np.float)
        y = np.asarray(data_y, dtype=np.int)
        return x, y

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    return train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y


def load_data(n_frame: int = 1):

    print('>>>>>>> loading data')
    cf = ConfigParser()
    cf.read('config.ini')

    frame_postfix = '_' + str(n_frame) + 'f'

    x_train = cf.get('dataset', 'x_train' + frame_postfix)
    y_train = cf.get('dataset', 'y_train' + frame_postfix)
    x_valid = cf.get('dataset', 'x_valid' + frame_postfix)
    y_valid = cf.get('dataset', 'y_valid' + frame_postfix)
    x_test = cf.get('dataset', 'x_test' + frame_postfix)
    y_test = cf.get('dataset', 'y_test' + frame_postfix)

    rval = [x_train, y_train, x_valid, y_valid, x_test, y_test]

    for i in range(len(rval)):
        with open(rval[i], 'rb') as f:
            rval[i] = pickle.load(f)

    print('<<<<<<< load data done')

    return rval

def preprocess_data(n_frame: int):
    print('>>>>>>> preprocess data', n_frame, 'frame')
    cf = ConfigParser()
    cf.read('config.ini')

    frame = '_' + str(n_frame) + 'f'

    orig_str = ['x', 'y' + frame]
    orig = []
    for i in range(len(orig_str)):
        path = cf.get('dataset', orig_str[i])
        with open(path, 'rb') as f:
            orig.append(pickle.load(f))
    x, y = orig[0], orig[1]

    print(x.shape, y.shape)

    x_train, y_train, x_valid, y_valid, x_test, y_test = [], [], [], [], [], []
    train_valid, valid_test = 0.6, 0.8
    dataset_size = y.shape[0]

    if n_frame == 1:
        for i in range(dataset_size):
            gray = np.array(cv2.cvtColor(x[i], cv2.COLOR_BGR2GRAY), dtype=np.float)
            gray /= 255.0

            if i < dataset_size * train_valid:
                x_train.append(gray)
                y_train.append(y[i])
            elif i < dataset_size * valid_test:
                x_valid.append(gray)
                y_valid.append(y[i])
            else:
                x_test.append(gray)
                y_test.append(y[i])
    elif n_frame == 2:
        for i in range(dataset_size):
            gray_1 = (np.array(cv2.cvtColor(x[i], cv2.COLOR_BGR2GRAY), dtype=np.float) / 255.0).ravel()
            gray_2 = (np.array(cv2.cvtColor(x[i + 1], cv2.COLOR_BGR2GRAY), dtype=np.float) / 255.0).ravel()
            gray = np.concatenate((gray_1, gray_2))
            if i < dataset_size * train_valid:
                x_train.append(gray)
                y_train.append(y[i])
            elif i < dataset_size * valid_test:
                x_valid.append(gray)
                y_valid.append(y[i])
            else:
                x_test.append(gray)
                y_test.append(y[i])

    x_train = np.array(x_train, dtype=np.float)
    x_valid = np.array(x_valid, dtype=np.float)
    x_test = np.array(x_test, dtype=np.float)
    y_train = np.array(y_train, dtype=np.int)
    y_valid = np.array(y_valid, dtype=np.int)
    y_test = np.array(y_test, dtype=np.int)

    index = random.sample(range(x_train.shape[0]), x_train.shape[0])
    x_train = x_train[index]
    y_train = y_train[index]

    print(x_train.shape, y_train.shape)

    vars_str = ['x_train' + frame, 'y_train' + frame, 'x_valid' + frame, 'y_valid' + frame, 'x_test' + frame, 'y_test' + frame]
    vars = [x_train, y_train, x_valid, y_valid, x_test, y_test]

    for i in range(len(vars)):
        path = cf.get('dataset', vars_str[i])
        with open(path, 'wb') as f:
            pickle.dump(vars[i], f)

    print('<<<<<<< preprocess data done')

def main():

    # preprocess_data(2)

    x_train, y_train, x_validation, y_validation, x_test, y_test = load_data(2)

    print(x_train.shape, np.sum(y_train > 0))
    print(x_validation.shape, np.sum(y_validation > 0))
    print(x_test.shape, np.sum(y_test > 0))
    #
    # # hiddens = [[2], [5], [10], [20]]
    # hiddens = [[2], [5]]
    # print('>>>>>>> start training')
    # for hidden in hiddens:
    #     tag = {'hd': len(hidden), 'hn': list_to_string(hidden, partition='_')}
    #     raw = '_raw'
    #     hog = '_hog'
    #     label = 'MLP' + dict_to_string(tag, ip='_', op='_', prefix='_')
    #     print('>>>>>>>', label)
    #     # classifier = MLP(n_in=x_train.shape[1], n_hidden=hidden, n_out=2)
    #     #
    #     # classifier.train(x_train.transpose(), y_train, x_validation.transpose(), y_validation,
    #     #                  n_epochs=1000, batch_size=200, patience=240000,
    #     #                  learning_rate=0.1, reg_lambda=0.0001)
    #     # save_mlp(classifier, dir='model')
    #     # classifier = load_mlp(generate_filename(dir='model', basename='mlp', tag=tag, postfix='pkl'))
    #     #
    #     # print('zero-one loss on validation dataset',
    #     #       round(classifier.compute_loss(x_validation.transpose(), y_validation, mode='zero-one') * 100, 2), '%')
    #     # print('zero-one loss on test dataset',
    #     #       round(classifier.compute_loss(x_test.transpose(), y_test, mode='zero-one') * 100, 2), '%')
    #     # prob = classifier.feed_forward(x_test.transpose())
    #     # plots = get_roc_points(prob[1], y_test)
    #     # with open('points'+raw+'/'+label+'_80000.pkl', 'wb') as f:
    #     #     pickle.dump((plots[:, 0], plots[:, 1]), f)
    #     # plt.plot(plots[:, 0], plots[:, 1], label=label+raw)
    #
    #     with open('points'+raw+'/'+label+'_80000.pkl', 'rb') as f:
    #         plots_x, plots_y = pickle.load(f)
    #     plt.plot(plots_x, plots_y, label=label)
    #
    #     print('<<<<<<< end', label)
    # print('<<<<<<< end training')
    #
    #
    # label = 'SVM_c_32_g_0.01_hog_80000'
    # with open('../SVM/points/' + label + '.pkl', 'rb') as f:
    #     plots_x, plots_y = pickle.load(f)
    # plt.plot(plots_x, plots_y, label=label)
    #
    # # label = 'SVM_c_32_g_0.01_raw'
    # # with open('../SVM/points/' + label + '.pkl', 'rb') as f:
    # #     plots_x, plots_y = pickle.load(f)
    # # plt.plot(plots_x, plots_y, label=label)
    #
    # label = 'LR_0.1_hog_80000'
    # with open('../LogisticRegression/points/' + label + '.pkl', 'rb') as f:
    #     plots_x, plots_y = pickle.load(f)
    # plt.plot(plots_x, plots_y, label=label)
    #
    # # label = 'LR_0.1_raw'
    # # with open('../LogisticRegression/points/' + label + '.pkl', 'rb') as f:
    # #     plots_x, plots_y = pickle.load(f)
    # # plt.plot(plots_x, plots_y, label=label)
    #
    # plt.xscale('log')
    # plt.xlabel('false alarm')
    # plt.ylabel('missing rate')
    # plt.title('ROC curve')
    # plt.grid()
    # plt.legend()
    # plt.show()

if __name__ == '__main__':

    main()


