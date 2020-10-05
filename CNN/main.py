import gzip
import random
import cv2
from configparser import ConfigParser

from cnn import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from cz.data import *

from mlp import *


def load_mnist_data():
    with gzip.open('mnist.pkl.gz', 'rb') as f:
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
            gray_1 = (np.array(cv2.cvtColor(x[i], cv2.COLOR_BGR2GRAY), dtype=np.float) / 255.0)
            gray_2 = (np.array(cv2.cvtColor(x[i + 1], cv2.COLOR_BGR2GRAY), dtype=np.float) / 255.0)
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

class CNN:

    def __init__(self):

        self.convolve_layer0 = ConvolutionLayer(
            in_shape=(1, 96, 48),
            filter_shape=(4, 1, 5, 5)
        )
        self.pool_layer0 = PoolLayer(
            in_shape=(4, 92, 44),
            pool_shape=(2, 2)
        )
        self.convolve_layer1 = ConvolutionLayer(
            in_shape=(4, 46, 22),
            filter_shape=(6, 4, 5, 5)
        )
        self.pool_layer1 = PoolLayer(
            in_shape=(6, 42, 18),
            pool_shape=(2, 2)
        )
        self.mlp = MLP(
            n_in=6 * 21 * 9,
            n_hidden=[500],
            n_out=2
        )

    def feedforward(self, x: np.ndarray):
        output0 = self.convolve_layer0.compute_output(x)
        output1 = self.pool_layer0.compute_output(output0)
        output2 = self.convolve_layer1.compute_output(output1)
        output3 = self.pool_layer1.compute_output(output2)
        output3 = output3.reshape(output3.shape[0], np.product(output3.shape[1:]))
        output4 = self.mlp.feed_forward(output3.transpose())
        return output4

    def backpropagation(self, x: np.ndarray, y: np.ndarray, learning_rate: np.float32):
        prev_x = self.pool_layer1.a
        prev_x = self.pool_layer1.a.reshape(prev_x.shape[0], np.product(prev_x.shape[1:]))
        next_delta, next_W = self.mlp.back_propagation(prev_x.transpose(), y, learning_rate, reg_lambda=0)
        next_delta = next_delta.transpose()
        next_delta = next_delta.reshape(*next_delta.shape, 1, 1)
        next_K = next_W.transpose().reshape(next_W.shape[1], *self.pool_layer1.out_shape)

        next_delta = self.pool_layer1.compute_delta(next_delta, next_K)
        next_beta = self.pool_layer1.beta

        next_delta = next_delta.repeat(self.pool_layer1.pool_shape[0], axis=2).repeat(self.pool_layer1.pool_shape[1], axis=3) * self.pool_layer1.up
        next_delta = self.convolve_layer1.compute_delta(next_delta, next_beta)
        next_K = self.convolve_layer1.K
        self.convolve_layer1.update_K_b(self.pool_layer0.a, learning_rate)

        next_delta = self.pool_layer0.compute_delta(next_delta, next_K)
        next_beta = self.pool_layer0.beta

        next_delta = next_delta.repeat(self.pool_layer0.pool_shape[0], axis=2).repeat(self.pool_layer0.pool_shape[1], axis=3) * self.pool_layer0.up
        next_delta = self.convolve_layer0.compute_delta(next_delta, next_beta)
        self.convolve_layer0.update_K_b(x, learning_rate)

    def train(self, x: np.ndarray, y: np.ndarray, x_validation: np.ndarray, y_validation: np.ndarray, patience: int,
              n_epochs: np.int, batch_size: np.int, learning_rate: np.float32):
        assert n_epochs > 0 and batch_size > 0 and learning_rate > 0

        dataset_size = x.shape[0]
        n_train_batches = dataset_size // batch_size

        # early-stopping parameters
        rest_patience = patience  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
        improvement_threshold = 0.995  # a relative improvement of this much is
        validation_frequency = min(n_train_batches, patience // 2)
        best_validation_loss, best_iter = np.inf, 0

        epoch = 0
        done_loop = False
        pb = ProgressBar(n_epochs, mode='fraction', save_path='pb_status_3')
        pb.add_status([5, 10])
        pb_patience, pb_validation_loss = str(rest_patience), ''
        pb.start()
        while (epoch < n_epochs) and not done_loop:
            epoch += 1
            for batch_index in range(n_train_batches):
                begin = batch_index * batch_size
                end = (batch_index + 1) * batch_size
                self.feedforward(x[begin:end])
                self.backpropagation(x[begin:end], y[begin:end], learning_rate)

                iter = (epoch - 1) * n_train_batches + batch_index
                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    this_validation_loss = self.compute_loss(x_validation, y_validation, mode='zero-one')
                    pb_validation_loss = str(round(this_validation_loss * 100, 5)) + '%'

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        # improve patience if loss improvement is good enough
                        if (this_validation_loss < best_validation_loss * improvement_threshold):
                            d = patience - rest_patience
                            patience = max(patience, iter * patience_increase)
                            rest_patience = patience - d

                        best_validation_loss = this_validation_loss
                        best_iter = iter
                rest_patience -= 1

                if patience <= iter:
                    done_loop = True
                    break
            pb.show_with(epoch, [str(rest_patience), pb_validation_loss])
        pb.end()

    def predict(self, x: np.ndarray):
        probability = self.feedforward(x)
        return np.argmax(probability, axis=0)   # (dataset_size, )

    def compute_loss(self, x: np.ndarray, y: np.ndarray, mode: str = None):
        if mode is None:
            mode = 'nll'
        if mode == 'nll':
            dataset_size = y.shape[0]
            loss = -np.mean(
                np.log(self.feedforward(x)[y, np.arange(dataset_size)]))
        elif mode == 'zero-one':
            temp = abs(self.predict(x) - y)
            loss = np.mean(temp != 0)
        else:
            loss = self.compute_loss(x, y, mode=None)

        return loss


def main():
    # # preprocess_data(2)
    #
    # # x_train, y_train, x_valid, y_valid, x_test, y_test = load_mnist_data()
    # x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(2)
    # print(x_train.shape, y_train.shape)
    #
    # # x_train = x_train
    # # y_train = y_train
    #
    # x_train = x_train.reshape((x_train.shape[0], 1, 96, 48))
    # x_valid = x_valid.reshape((x_valid.shape[0], 1, 96, 48))
    # x_test = x_test.reshape((x_test.shape[0], 1, 96, 48))
    #
    # classifier = CNN()
    # classifier.train(x_train, y_train, x_valid, y_valid, 100000, 100, 500, np.float32(0.1))
    # path = generate_filename(dir='model', basename='cnn', postfix='pkl')
    # with open(path, 'wb') as f:
    #     pickle.dump(classifier, f)
    #
    # prob = classifier.feedforward(x_test)
    # plots = get_roc_points(prob[1], y_test)

    # with open('points.pkl', 'wb') as f:
    #     pickle.dump((plots[:, 0], plots[:, 1]), f)
    # plt.plot(plots[:, 0], plots[:, 1], label='CNN')

    label = 'CNN'
    with open('points.pkl', 'rb') as f:
        plots_x, plots_y = pickle.load(f)
    plt.plot(plots_x, plots_y, label=label)

    label = 'SVM_c_32_g_0.01_hog'
    with open('../SVM/points/' + label + '.pkl', 'rb') as f:
        plots_x, plots_y = pickle.load(f)
    plt.plot(plots_x, plots_y, label=label)

    label = 'MLP_hd_1_hn_20'
    with open('../MLP/points_raw/' + label + '.pkl', 'rb') as f:
        plots_x, plots_y = pickle.load(f)
    plt.plot(plots_x, plots_y, label=label)

    plt.xscale('log')
    plt.xlabel('false alarm')
    plt.ylabel('missing rate')
    plt.title('ROC curve')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':

    main()
    # f()
    # a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # # print(np.argmax(a))
    # b = np.array([[1, 2], [3, 4]])
    # sw = Stopwatch()
    # sw.start()
    # signal.convolve2d(a, b, 'valid')
    # sw.lap()
    # print(sw.elapse[-1])
    # a_fft = fftpack.fft2(a, (4, 4))
    # b_fft = fftpack.fft2(b, (4, 4))
    # np.real(fftpack.ifft2(a_fft * b_fft))[1:-1, 1:-1]
    # sw.lap()
    # print(sw.elapse[-1])

    # print()
    # print(signal.convolve2d(a, np.rot90(b, 2)))




