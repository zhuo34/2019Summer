import pickle

import numpy as np
from cz.progressbar import *
from cz.string import *

class MLP:
    @property
    def n_in(self):
        return self.__n_in

    @n_in.setter
    def n_in(self, n_in: int):
        assert n_in > 0
        self.__n_in = n_in

    @property
    def n_out(self):
        return self.__n_out

    @n_out.setter
    def n_out(self, n_out: int):
        assert n_out > 0
        self.__n_out = n_out

    @property
    def n_hidden(self):
        return self.__n_hidden

    @n_hidden.setter
    def n_hidden(self, n_hidden: list):
        # assert len(n_hidden) > 0
        self.__n_hidden = n_hidden

    def check_shape_x(self, x: np.ndarray):
        assert x.ndim == 2 and x.shape[0] == self.n_in

    def check_shape_x_y(self, x: np.ndarray, y: np.ndarray):
        self.check_shape_x(x)
        assert y.ndim == 1 and y.shape[0] == x.shape[1]

    def __init__(self, n_in: int, n_hidden: list, n_out: int, rng: np.random.RandomState = None):
        self.__n_in = n_in
        self.n_hidden = n_hidden
        self.__n_out = n_out

        self.hidden_layers = []
        for i in range(len(self.n_hidden)):
            if i == 0:
                this_n_in = self.n_in
            else:
                this_n_in = self.n_hidden[i - 1]
            this_n_out = self.n_hidden[i]
            hidden_layer = HiddenLayer(
                n_in=this_n_in,
                n_out=this_n_out,
                activation='tanh',
                rng=rng
            )
            self.hidden_layers.append(hidden_layer)
        if len(self.n_hidden) == 0:
            this_n_in = self.n_in
        else:
            this_n_in = n_hidden[-1]
        self.output_layer = OutputLayer(
            n_in=this_n_in,
            n_out=self.n_out
        )

    def feed_forward(self, x: np.ndarray):
        for layer in self.hidden_layers:
            layer.compute_output(x)
            x = layer.a
        return self.output_layer.compute_output(x)

    def back_propagation(self, x, y, learning_rate: np.float, reg_lambda: np.float):
        self.output_layer.compute_delta(y)
        if len(self.n_hidden) == 0:
            prev_x = x
        else:
            prev_x = self.hidden_layers[-1].a
        next_W = self.output_layer.W.copy()
        self.output_layer.update_W_b(prev_x, learning_rate, reg_lambda)
        next_delta = self.output_layer.delta
        for i in reversed(range(len(self.n_hidden))):
            curr_layer = self.hidden_layers[i]
            curr_layer.compute_delta(next_W, next_delta)
            if i > 0:
                prev_x = self.hidden_layers[i - 1].a
            else:
                prev_x = x
            next_W = curr_layer.W.copy()
            curr_layer.update_W_b(prev_x, learning_rate, reg_lambda)
            next_delta = curr_layer.delta

        return next_delta, next_W

    def train(self, x: np.ndarray, y: np.ndarray, x_validation: np.ndarray, y_validation: np.ndarray, patience: int,
              n_epochs: np.int, batch_size: np.int, learning_rate: np.float, reg_lambda: np.float):
        self.check_shape_x_y(x, y)
        self.check_shape_x_y(x_validation, y_validation)
        assert n_epochs > 0 and batch_size > 0 and learning_rate > 0 and reg_lambda >= 0

        dataset_size = x.shape[1]
        n_train_batches = dataset_size // batch_size

        # early-stopping parameters
        rest_patience = patience  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
        improvement_threshold = 0.995  # a relative improvement of this much is
        validation_frequency = min(n_train_batches, patience // 2)
        best_validation_loss, best_iter = np.inf, 0

        epoch = 0
        done_loop = False
        pb = ProgressBar(n_epochs, mode='fraction', save_path='pb_status')
        pb.add_status([5, 10])
        pb_patience, pb_validation_loss = str(rest_patience), ''
        pb.start()
        while (epoch < n_epochs) and not done_loop:
            epoch += 1
            for batch_index in range(n_train_batches):
                begin = batch_index * batch_size
                end = (batch_index + 1) * batch_size
                self.feed_forward(x[:, begin:end])
                self.back_propagation(x[:, begin:end], y[begin:end], learning_rate, reg_lambda)


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
        self.check_shape_x(x)

        probability = self.feed_forward(x)
        return np.argmax(probability, axis=0)   # (dataset_size, )

    def compute_loss(self, x: np.ndarray, y: np.ndarray, mode: str = None, reg_lambda: np.float = 0):
        self.check_shape_x_y(x, y)

        if mode is None:
            mode = 'nll'
        if mode == 'nll':
            dataset_size = y.shape[0]
            loss = -np.mean(
                np.log(self.feed_forward(x)[y, np.arange(dataset_size)]))
            if reg_lambda > 0:
                for i in range(len(self.n_hidden)):
                    loss += np.sum(self.hidden_layers[i].W ** 2)
                loss += np.sum(self.output_layer.W ** 2)
        elif mode == 'zero-one':
            temp = abs(self.predict(x) - y)
            loss = np.mean(temp != 0)
        else:
            loss = self.compute_loss(x, y, mode=None, reg_lambda=reg_lambda)

        return loss

def save_mlp(mlp: MLP, dir: str = None):
    tag = {'hd': len(mlp.n_hidden), 'hn': list_to_string(mlp.n_hidden, partition='_')}
    path = generate_filename(dir=dir, basename='mlp', tag=tag, postfix='pkl')
    with open(path, 'wb') as f:
        pickle.dump(mlp, f)
    return path

def load_mlp(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


class MLPLayer:

    @property
    def n_in(self):
        return self.__n_in

    @n_in.setter
    def n_in(self, n_in: int):
        assert n_in > 0
        self.__n_in = n_in

    @property
    def n_out(self):
        return self.__n_out

    @n_out.setter
    def n_out(self, n_out: int):
        assert n_out > 0
        self.__n_out = n_out

    @property
    def W(self):
        return self.__W

    @W.setter
    def W(self, W: np.ndarray):
        self.check_shape_W(W)
        self.__W = W

    @property
    def b(self):
        return self.__b

    @b.setter
    def b(self, b: np.ndarray):
        if b.ndim == 1:
            b = b.reshape(1, b.shape[0])
        self.check_shape_b(b)
        self.__b = b

    def check_shape_x(self, x: np.ndarray):
        assert x.ndim == 2 and x.shape[0] == self.n_in

    def check_shape_x_y(self, x: np.ndarray, y: np.ndarray):
        self.check_shape_x(x)
        assert y.ndim == 1 and y.shape[0] == x.shape[1]

    def check_shape_W(self, W: np.ndarray):
        assert W.shape == (self.n_in, self.n_out)

    def check_shape_b(self, b: np.ndarray):
        assert b.shape == (1, self.n_out)

    def check_shape_W_b(self, W: np.ndarray, b: np.ndarray):
        self.check_shape_W(W)
        self.check_shape_b(b)

    def check_shape_gradient_W_b(self, gradient_W: np.ndarray, gradient_b: np.ndarray):
        assert self.W.transpose().shape == gradient_W.shape
        assert self.b.transpose().shape == gradient_b.shape

    def __init__(self, n_in: int, n_out: int):
        '''
        :param input: data for training where one column represents one sample
        :param n_out: class dimension
        :param activation: activation function
        '''
        self.n_in = n_in
        self.n_out = n_out

        # W is a matrix where column-k represents the separation hyperplane for class-k
        self.W = np.zeros((self.n_in, self.n_out), dtype=np.float)

        # b is a vector where column-k represents the free parameter for class-k
        self.b = np.zeros((1, self.n_out), dtype=np.float)


class HiddenLayer(MLPLayer):

    @property
    def activation(self):
        return self.__activation

    @activation.setter
    def activation(self, activation: str):
        if activation is None:
            activation = self.activation_functions[0]
        elif activation not in self.activation_functions:
            activation = self.activation_functions[0]
        self.__activation = activation

    @property
    def activation_functions(self):
        return ['tanh', 'sigmoid']

    def __init__(self, n_in: int, n_out: int, activation: str = 'tanh', rng: np.random.RandomState = None):
        super().__init__(n_in, n_out)

        self.activation = activation

        # W is a matrix where column-k represents the separation hyperplane for class-k
        # b is a vector where column-k represents the free parameter for class-k
        if rng is None:
            rng = np.random.RandomState(int(time.time()))
        if self.activation == 'tanh':
            self.W = np.asarray(rng.uniform(
                low=-np.sqrt(6. / (self.n_in + self.n_out)),
                high=np.sqrt(6. / (self.n_in + self.n_out)),
                size=(self.n_in, self.n_out)
            ), dtype=np.float)
        elif self.activation == 'sigmoid':
            self.W = np.asarray(rng.uniform(
                low=-np.sqrt(6. / (self.n_in + self.n_out)) * 4,
                high=np.sqrt(6. / (self.n_in + self.n_out)) * 4,
                size=(self.n_in, self.n_out)
            ), dtype=np.float)


    def compute_delta(self, next_W: np.ndarray, next_delta: np.ndarray):
        assert next_W.ndim == 2 and self.n_out == next_W.shape[0]
        assert next_delta.ndim == 2 and next_W.shape[1] == next_delta.shape[0]

        mul1 = np.dot(next_W, next_delta)       # n_out * dataset_size
        mul2 = 0
        if self.activation == 'tanh':
            mul2 = 1 - self.a ** 2
        elif self.activation == 'sigmoid':
            mul2 = self.a * (1 - self.a)
        self.delta = mul1 * mul2                # n_out * dataset_size
        return self.delta

    def compute_output(self, x: np.ndarray):
        self.check_shape_x(x)
        output = np.dot(self.W.transpose(), x) + self.b.transpose()
        if self.activation == 'tanh':
            output = np.tanh(output)
        elif self.activation == 'sigmoid':
            output = 1 / (1 + np.exp(-output))
        self.a = output
        self.dataset_size = self.a.shape[1]
        return output   # n_out * dataset_size

    def update_W_b(self, x: np.ndarray, learning_rate: np.float, reg_lambda: np.float):
        self.check_shape_x(x)
        self.W -= learning_rate * (np.dot(x, self.delta.transpose()) / self.dataset_size + reg_lambda * self.W)
        self.b -= learning_rate * (self.delta.mean(axis=1))


# softmax
class OutputLayer(MLPLayer):

    def __init__(self, n_in: int, n_out: int):
        super().__init__(n_in, n_out)

    def compute_probability(self, x: np.ndarray):
        self.check_shape_x(x)
        z = np.dot(self.W.transpose(), x) + self.b.transpose()
        exp_z = np.exp(z)
        sum = np.sum(exp_z, axis=0)
        return exp_z / sum  # n_out * dataset_size

    def compute_delta(self, y: np.ndarray):
        assert y.ndim == 1 and y.shape[0] == self.dataset_size

        y_match = np.arange(self.n_out).reshape((self.n_out, 1)) == y
        self.delta = self.a - y_match           # n_out * dataset_size
        return self.delta

    def compute_output(self, x: np.ndarray):
        self.a = self.compute_probability(x)
        self.dataset_size = self.a.shape[1]
        return self.a

    def update_W_b(self, x: np.ndarray, learning_rate: np.float, reg_lambda: np.float):
        self.check_shape_x(x)
        self.W -= learning_rate * (np.dot(x, self.delta.transpose()) / self.dataset_size + reg_lambda * self.W)
        self.b -= learning_rate * (self.delta.mean(axis=1)).transpose()
