from keras.utils import to_categorical
from keras.datasets import mnist
from keras import utils
from GAModel.DataMgr import *
from GAModel.SteadyStateGAModel import SteadyStateGA
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

X_train, y_train, X_test, y_test = load_mnist()
# (X_train, Y_train), (X_test, Y_test) = mnist.load_data(path=path)
# X_train = X_train.reshape(X_train.shape[0], 28, 28, 1) / 255
# X_test = X_test.reshape(X_test.shape[0], 28, 28, 1) / 255
# y_train = utils.to_categorical(Y_train, num_classes=10)
# y_test = utils.to_categorical(Y_test, num_classes=10)
train_size = len(X_train)
test_size = len(X_test)

g = SteadyStateGA(
    _X_train=X_train[:train_size],
    _y_train=y_train[:train_size],
    _X_test=X_test[:test_size],
    _y_test=y_test[:test_size],
    _pop_size=20,
    _r_mutation=0.1,
    _p_crossover=0.7,
    _p_mutation=0,  # no use
    _max_iter=10,
    _min_fitness=0.95,
    _batch_size=5000,
)
g.run()

g.run()

write_performance(g.evaluation_history, 'SteadyStateGA_MNIST.txt')
