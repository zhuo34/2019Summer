import gzip
from configparser import ConfigParser
import os

from cz.progressbar import *
from cz.data import *

import pickle
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import cv2
import random


class LogisticRegression:

    @property
    def n_in(self):
        return self.__n_in

    @property
    def n_out(self):
        return self.__n_out

    @property
    def W(self):
        return self.__W

    @W.setter
    def W(self, W: np.ndarray):
        self.check_dim_W(W)
        self.__W = W

    @property
    def b(self):
        return self.__b

    @b.setter
    def b(self, b:np.ndarray):
        if b.ndim == 1:
            b = b.reshape(1, b.shape[0])
        self.check_dim_b(b)
        self.__b = b

    def __init__(self, n_in: int, n_out: int):
        '''
        :param input: data for training where one column represents one sample
        :param n_out: class dimension
        '''
        assert n_in > 0 and n_out > 1
        self.__n_in = n_in
        self.__n_out = n_out

        # W is a matrix where column-k represents the separation hyperplane for class-k
        self.W = np.zeros((self.n_in, self.n_out), dtype=np.float)

        # b is a vector where column-k represents the free parameter for class-k
        self.b = np.zeros((1, self.n_out), dtype=np.float)

    def check_dim_x(self, x: np.ndarray):
        assert x.ndim == 2 and x.shape[0] == self.n_in

    def check_dim_x_y(self, x: np.ndarray, y: np.ndarray):
        assert x.ndim == 2 and x.shape[0] == self.n_in
        assert y.ndim == 1 and y.shape[0] == x.shape[1]

    def check_dim_W(self, W: np.ndarray):
        assert W.shape == (self.n_in, self.n_out)

    def check_dim_b(self, b: np.ndarray):
        assert b.shape == (1, self.n_out)

    def check_dim_W_b(self, W: np.ndarray, b: np.ndarray):
        self.check_dim_W(W)
        self.check_dim_b(b)

    def check_dim_gradient_W_b(self, gradient_W: np.ndarray, gradient_b: np.ndarray):
        assert self.W.transpose().shape == gradient_W.shape
        assert self.b.transpose().shape == gradient_b.shape

    def compute_probability(self, x: np.ndarray):
        self.check_dim_x(x)
        z = np.dot(self.W.transpose(), x) + self.b.transpose()
        exp_z = np.exp(z)
        sum = np.sum(exp_z, axis=0)
        return exp_z / sum  # n_out * dataset_size

    def predict(self, x: np.ndarray):
        self.check_dim_x(x)

        probability = self.compute_probability(x)
        return np.argmax(probability, axis=0)  # (dataset_size, )

    def compute_loss(self, x: np.ndarray, y: np.ndarray, regulation_lambda: float = 0, mode: str = None):
        self.check_dim_x_y(x, y)
        norm = np.sum(self.W ** 2) + np.sum(self.b ** 2)
        if mode is None:
            return self.compute_loss(x, y, regulation_lambda, mode='nll')
        elif mode == 'nll':
            dataset_size = y.shape[0]
            return -np.mean(np.log(self.compute_probability(x)[y, np.arange(dataset_size)])) + regulation_lambda * norm / 2
        elif mode == 'zero-one':
            temp = abs(self.predict(x) - y)
            return np.mean(temp != 0) + regulation_lambda * norm / 2
        else:
            return self.compute_loss(x, y, regulation_lambda, mode=None)

    def compute_gradient(self, x: np.ndarray, y: np.ndarray, regulation_lambda: float = 0):
        self.check_dim_x_y(x, y)
        dataset_size = y.shape[0]

        y_match = np.arange(self.n_out).reshape((self.n_out, 1)) == y
        coef = y_match - self.compute_probability(x)

        gradient_W = -(np.dot(coef, x.transpose()) / dataset_size) + regulation_lambda * self.W.transpose()
        gradient_b = -coef.mean(axis=1).reshape(self.n_out, 1) + regulation_lambda * self.b.transpose()

        return gradient_W, gradient_b  # n_out * n_in, n_out * 1

    def update_gradient(self, gradient_W: np.ndarray, gradient_b: np.ndarray, learning_rate: float):
        self.check_dim_gradient_W_b(gradient_W, gradient_b)
        assert learning_rate > 0
        self.W -= learning_rate * gradient_W.transpose()
        self.b -= learning_rate * gradient_b.transpose()

    def update_gradient_line_search(self, gradient_W: np.ndarray, gradient_b: np.ndarray, learning_rate: float, x: np.ndarray, y: np.ndarray):
        self.check_dim_gradient_W_b(gradient_W, gradient_b)
        assert learning_rate > 0

        new_classifier = LogisticRegression(self.n_in, self.n_out)
        new_classifier.W = self.W.copy()
        new_classifier.b = self.b.copy()
        norm = np.sum(gradient_W ** 2, axis=1) + gradient_b ** 2  # n_out * 1

        alpha, beta = 0.5, 0.5

        for i in range(self.n_out):
            this_learning_rate = 4.0
            loss = self.compute_loss(x, y)
            new_classifier.W[:, i] = self.W[:, i] - this_learning_rate * gradient_W.transpose()[:, i]
            new_classifier.b[0, i] = self.b[0, i] - this_learning_rate * gradient_b.transpose()[0, i]
            while True:
                if new_classifier.compute_loss(x, y) <= loss - alpha * this_learning_rate * norm[i, 0]:
                    break
                this_learning_rate *= beta
                if this_learning_rate < learning_rate:
                    this_learning_rate = learning_rate
                    break
                new_classifier.W[:, i] = self.W[:, i] - this_learning_rate * gradient_W.transpose()[:, i]
                new_classifier.b[0, i] = self.b[0, i] - this_learning_rate * gradient_b.transpose()[0, i]
            self.W[:, i] -= this_learning_rate * gradient_W.transpose()[:, i]
            self.b[0, i] -= this_learning_rate * gradient_b.transpose()[0, i]

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray = None,
              y_validation: np.ndarray = None, line_search: bool = False, patience: int = None,
              learning_rate: float = 0.1, batch_size: int = None, n_epochs: int = 1000, regulation_lambda: float = 0, save: bool = False):
        self.check_dim_x_y(x_train, y_train)

        assert learning_rate > 0

        assert regulation_lambda >= 0
        assert n_epochs > 0

        has_validation = False
        if (x_validation is not None) or (y_validation is not None):
            assert (x_validation is not None) and (y_validation is not None)
            self.check_dim_x_y(x_validation, y_validation)
            has_validation = True

        if batch_size is None:
            batch_size = x_train.shape[1]
        assert batch_size > 0

        n_train_batches = x_train.shape[1] // batch_size
        if patience is None:
            patience = n_train_batches * 1000
        patience_increase = 2
        improvement_threshold = 0.995
        validation_frequency = min(n_train_batches, patience // 2)
        best_validation_loss = np.inf

        loop_done = False
        epoch = 0
        pb = ProgressBar(n_epochs, n_step=50, mode='fraction')
        pb.start()
        pb.add_status([6, 10, 10])
        current_patience, pb_train_loss, pb_validation_loss = patience, '', ''
        validation_losses = []
        this_validation_loss = self.compute_loss(x_validation, y_validation, regulation_lambda)
        validation_losses.append(this_validation_loss)
        while (epoch < n_epochs) and (not loop_done):
            epoch += 1
            for batch_index in range(n_train_batches):
                begin_index = batch_index * batch_size
                end_index = (batch_index + 1) * batch_size

                gradient_W, gradient_b = self.compute_gradient(x_train[:, begin_index: end_index],
                                                               y_train[begin_index: end_index], regulation_lambda)
                if line_search:
                    self.update_gradient_line_search(gradient_W, gradient_b, learning_rate, x_train, y_train)
                else:
                    self.update_gradient(gradient_W, gradient_b, learning_rate)

                iter = (epoch - 1) * n_train_batches + batch_index
                if has_validation:
                    if (iter + 1) % validation_frequency == 0:

                        this_train_loss = self.compute_loss(x_train, y_train, regulation_lambda)
                        pb_train_loss = str(round(this_train_loss, 6))

                        this_validation_loss = self.compute_loss(x_validation, y_validation, regulation_lambda, mode='zero-one')
                        pb_validation_loss = str(round(this_validation_loss * 100, 6)) + '%'

                        if this_validation_loss < best_validation_loss:
                            if this_validation_loss < best_validation_loss * improvement_threshold:
                                d = patience - current_patience
                                patience = max(patience, iter * patience_increase)
                                current_patience = patience - d
                            best_validation_loss = this_validation_loss
                    if patience <= iter:
                        loop_done = True
                        break
                else:
                    pass
                current_patience -= 1
            pb.show_with(epoch, [current_patience, pb_train_loss, pb_validation_loss])
        pb.end()

        if save:
            return self.save()

    def clear(self):
        self.W[:, :] = 0.
        self.b[:, :] = 0.


    def save(self, dir: str = None, tag: dict = None):
        path = generate_filename(dir, 'lr', tag, postfix='pkl')
        with open(path, 'wb') as f:
            pickle.dump(np.concatenate((self.W, self.b)), f)
        return path

    def load(self, path: str):
        with open(path, 'rb') as f:
            theta = pickle.load(f)
        assert theta.ndim == 2
        self.W = theta[:-1]
        self.b = theta[-1]

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
    x, y = orig[0][100000:180000], orig[1][100000:180000-1]

    print(x.shape, y.shape)

    x_train, y_train, x_valid, y_valid, x_test, y_test = [], [], [], [], [], []
    train_valid, valid_test = 0.6, 0.8
    dataset_size = y.shape[0]

    hog_descriptor = cv2.HOGDescriptor((48, 48), (16, 16), (8, 8), (8, 8), 9)

    if n_frame == 1:
        for i in range(dataset_size):
            hog = hog_descriptor.compute(np.array(x[i]) / 255.0)
            if i < dataset_size * train_valid:
                x_train.append(hog)
                y_train.append(y[i])
            elif i < dataset_size * valid_test:
                x_valid.append(hog)
                y_valid.append(y[i])
            else:
                x_test.append(hog)
                y_test.append(y[i])
    elif n_frame == 2:
        for i in range(dataset_size):
            hog_1 = hog_descriptor.compute(x[i])
            hog_2 = hog_descriptor.compute(x[i + 1])
            hog = np.concatenate((hog_1, hog_2), axis=0)
            # print(hog.shape)
            if i < dataset_size * train_valid:
                x_train.append(hog)
                y_train.append(y[i])
            elif i < dataset_size * valid_test:
                x_valid.append(hog)
                y_valid.append(y[i])
            else:
                x_test.append(hog)
                y_test.append(y[i])

    x_dim = x_train[0].shape[0] * x_train[0].shape[1]
    x_train = np.array(x_train, dtype=np.float).reshape((len(x_train), x_dim))
    x_valid = np.array(x_valid, dtype=np.float).reshape((len(x_valid), x_dim))
    x_test = np.array(x_test, dtype=np.float).reshape((len(x_test), x_dim))
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

def load_mnist_data(dataset):
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    def process_data(data_xy):
        data_x, data_y = data_xy
        x = np.asarray(data_x, dtype=np.float)
        y = np.asarray(data_y, dtype=np.int)
        return x, y

    test_set_x, test_set_y = process_data(test_set)
    valid_set_x, valid_set_y = process_data(valid_set)
    train_set_x, train_set_y = process_data(train_set)

    return train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y

# def line_search_curve():
#     validation_loss_ls = load_pickle('./validation_loss_ls_1.pkl')
#     validation_loss_nls = load_pickle('./validation_loss_nls_1.pkl')
#     x = range(50)
#     print(len(validation_loss_ls))
#     print(len(validation_loss_nls))
#     plt.plot(x, validation_loss_ls[1:], '.-', label='line search')
#     plt.plot(x, validation_loss_nls[1:], '.-', label='naive')
#     plt.ylabel('validation loss')
#     plt.xlabel('epoch')
#     plt.title('convergence speed')
#     plt.show()


if __name__ == '__main__':

    # preprocess_data(2)
    x_train, y_train, x_validation, y_validation, x_test, y_test = load_data(2)

    # x_train, y_train, x_validation, y_validation, x_test, y_test = load_mnist_data('mnist.pkl.gz')

    print(x_train.shape, np.sum(y_train > 0))
    print(x_validation.shape, np.sum(y_validation > 0))
    print(x_test.shape, np.sum(y_test > 0))

    classifier = LogisticRegression(n_in=x_train.shape[1], n_out=2)

    classifier.fit(x_train.transpose(), y_train, x_validation.transpose(), y_validation, patience=None,
               n_epochs=1000, batch_size=200, learning_rate=0.1, regulation_lambda=0.0001)
    classifier.save('model')
    # classifier.load('model/lr.pkl')
    # print('zero-one loss on test dataset',
    #       round(classifier.compute_loss(x_test.transpose(), y_test, mode='zero-one') * 100, 2), '%')
    prob = classifier.compute_probability(x_test.transpose())
    plots = get_roc_points(prob[1], y_test)
    with open('points/LR_0.1_hog_80000.pkl', 'wb') as f:
        pickle.dump((plots[:, 0], plots[:, 1]), f)
    # with open('points/LR_0.1_hog.pkl', 'rb') as f:
    #     plots_x, plots_y = pickle.load(f)
    # plt.plot(plots_x, plots_y)
    plt.plot(plots[:, 0], plots[:, 1])
    # classifier.clear()
    plt.xscale('log')
    plt.xlabel('false alarm')
    plt.ylabel('missing rate')
    plt.title('ROC curve')
    plt.grid()
    plt.legend()
    plt.show()

