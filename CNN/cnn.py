import pickle

import numpy as np
from cz.progressbar import *
from cz.string import *
from scipy import signal, fftpack


class ConvolutionLayer:

    def __get_out_shape(self):
        out_width = self.in_shape[1] - self.filter_shape[2] + 1
        out_height = self.in_shape[2] - self.filter_shape[3] + 1
        return (self.filter_shape[0], out_width, out_height)

    def __check_filter_shape(self, in_shape: tuple, filter_shape: tuple):
        assert len(filter_shape) == 4 and len(in_shape) == 3
        assert filter_shape[1] == in_shape[0]
        assert filter_shape[2] <= in_shape[1]
        assert filter_shape[3] <= in_shape[2]

    def __check_x_shape(self, x: np.ndarray):
        assert x.ndim == 4 and x.shape[1:] == self.in_shape

    def __init__(self, in_shape: tuple, filter_shape: tuple):
        self.__check_filter_shape(in_shape, filter_shape)
        self.in_shape = in_shape
        self.filter_shape = filter_shape
        self.out_shape = self.__get_out_shape()

        fan_in = np.product(self.filter_shape[1:])
        fan_out = self.out_shape[0] * np.product(self.filter_shape[2:])
        rng = np.random.RandomState(int(time.time()))
        self.K = np.asarray(rng.uniform(
                        low=-np.sqrt(6. / (fan_in +fan_out)),
                        high=np.sqrt(6. / (fan_in +fan_out)),
                        size=self.filter_shape
                    ), dtype=np.float32)
        self.b = np.zeros((self.filter_shape[0], ), dtype=np.float32)

    def compute_output(self, x: np.ndarray):
        if x.ndim == 2:
            x = x.reshape(1, *x.shape)
        if x.ndim == 3:
            x = x.reshape(1, *x.shape)
        self.__check_x_shape(x)
        self.dataset_size = x.shape[0]

        self.a = np.zeros((self.dataset_size, *self.out_shape), dtype=np.float32)
        for data_index in range(self.dataset_size):
            for j in range(self.out_shape[0]):
                for i in range(self.in_shape[0]):
                    kernel = self.K[j][i]
                    self.a[data_index, j] += signal.correlate2d(x[data_index, i], kernel, 'valid')
                self.a[data_index, j] = np.tanh(self.a[data_index, j] + self.b[j])
        return self.a

    def compute_delta(self, next_delta: np.ndarray, beta: np.ndarray):
        self.delta = np.zeros((self.dataset_size, *self.out_shape), dtype=np.float32)
        for j in range(self.out_shape[0]):
            self.delta[:, j] = beta[j] * (next_delta[:, j] * (1 - (self.a[:, j]) ** 2))
        return self.delta

    def update_K_b(self, x: np.ndarray, learning_rate: np.float32):
        for j in range(self.out_shape[0]):
            self.b[j] -= learning_rate * (np.sum(self.delta[:, j]) / self.dataset_size)
            for i in range(self.in_shape[0]):
                gradient_K = np.zeros(self.filter_shape[2:], dtype=np.float32)
                for data_index in range(self.dataset_size):
                    gradient_K += signal.correlate2d(x[data_index, i], self.delta[data_index, j], 'valid')
                gradient_K /= self.dataset_size
                self.K[j, i] -= learning_rate * gradient_K


class PoolLayer:

    def __check_pool_shape(self, in_shape: tuple, pool_shape: tuple):
        assert len(in_shape) == 3 and len(pool_shape) == 2
        assert pool_shape[0] <= in_shape[1]
        assert pool_shape[1] <= in_shape[2]

    def __get_out_shape(self):
        out_width = self.in_shape[1] // self.pool_shape[0]
        out_height = self.in_shape[2] // self.pool_shape[1]
        return (self.in_shape[0], out_width, out_height)

    def __check_x_shape(self, x: np.ndarray):
        assert x.ndim == 4 and x.shape[1:] == self.in_shape

    def __init__(self, in_shape: tuple, pool_shape: tuple, mode: str = 'max'):
        self.__check_pool_shape(in_shape, pool_shape)
        self.in_shape = in_shape
        self.pool_shape = pool_shape
        self.out_shape = self.__get_out_shape()

        self.beta = np.ones((self.in_shape[0], ), dtype=np.float32)
        self.b = np.zeros((self.in_shape[0], ), dtype=np.float32)

    def compute_output(self, x: np.ndarray):
        if x.ndim == 2:
            x = x.reshape(1, *x.shape)
        if x.ndim == 3:
            x = x.reshape(1, *x.shape)
        self.__check_x_shape(x)
        self.dataset_size = x.shape[0]

        tt = x.reshape((self.dataset_size, self.out_shape[0], self.out_shape[1], self.pool_shape[0], self.out_shape[2], self.pool_shape[1])) \
            .swapaxes(3, 4).reshape(self.dataset_size, self.out_shape[0], -1, np.product(self.pool_shape))
        self.map = tt.max(3).reshape(self.dataset_size, self.out_shape[0], *self.out_shape[1:])
        max_index = tt.argmax(3)
        self.a = self.map.swapaxes(0, 1)
        for i in range(self.out_shape[0]):
            self.a[i] = self.beta[i] * self.a[i] + self.b[i]
        self.a = self.a.swapaxes(0, 1)

        self.up = max_index[..., None] == np.arange(np.product(self.pool_shape))
        self.up = self.up.reshape((self.dataset_size, self.out_shape[0], self.out_shape[1], self.out_shape[2], self.pool_shape[0], self.pool_shape[1]))\
            .swapaxes(3, 4).reshape((self.dataset_size, *self.in_shape)).astype(np.int)
        return self.a

    def compute_delta(self, next_delta: np.ndarray, next_K: np.ndarray):
        full_shape = get_convolve_shape2d(next_delta.shape[2:], next_K.shape[2:])
        next_delta_fft = fftpack.fft2(next_delta, full_shape)
        next_K_fft = fftpack.fft2(next_K, full_shape)
        self.delta = np.zeros((self.dataset_size, *self.out_shape), dtype=np.float32)
        for data_index in range(self.dataset_size):
            t = next_delta_fft[data_index].reshape(-1, 1, *full_shape)
            self.delta[data_index] = np.real(fftpack.ifft2(np.sum(t * next_K_fft, axis=0)))
        return self.delta

    def update_beta_b(self, learning_rate: np.float32):
        tt = self.delta * self.map
        for i in range(self.out_shape[0]):
            self.b[i] -= learning_rate * (np.sum(self.delta[:, i]) / self.dataset_size)
            self.beta[i] -= learning_rate * (np.sum(tt[:, i]) / self.dataset_size)


def get_convolve_shape2d(x_shape: tuple, kernel_shape: tuple, mode: str = 'full'):
    assert len(x_shape) == 2 and len(kernel_shape) == 2

    if mode == 'full':
        return (x_shape[0] + (kernel_shape[0] - 1), x_shape[1] + (kernel_shape[1] - 1))
    elif mode == 'valid':
        assert x_shape[0] >= kernel_shape[0] and x_shape[1] >= kernel_shape[1]
        return (x_shape[0] - (kernel_shape[0] - 1), x_shape[1] - (kernel_shape[1] - 1))
    else:
        return get_conv2d_shape(x_shape, kernel_shape)