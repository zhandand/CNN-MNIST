from keras import utils
from keras.datasets import mnist
import os

path = os.getcwd() + "\data\mnist.npz"


def load_mnist():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data(path=path)
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1) / 255
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1) / 255
    Y_train = utils.to_categorical(Y_train, num_classes=10)
    Y_test = utils.to_categorical(Y_test, num_classes=10)
    return X_train, Y_train, X_test, Y_test


def write_performance(perf, filename):
    with open(filename, mode='w') as f:
        text = 'iter\tbest_pop\tbest_loss\tbest_acc\tavg_fitness\n'
        for e in perf:
            text += ('\t'.join(list(map(str, [e['iter'], e['best_fit']['pop'], e['best_fit']['train_loss'],
                                              e['best_fit']['train_acc'], e['avg_fitness']]))) + '\n')
        f.write(text)
