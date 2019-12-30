import os
import matplotlib.pyplot as plt
from GAModel.DataMgr import *
from GAModel.SteadyStateGAModel import SteadyStateGA

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

X_train, y_train, X_test, y_test = load_mnist()

train_size = len(X_train)
test_size = len(X_test)

g = SteadyStateGA(
    _X_train=X_train[:train_size],
    _y_train=y_train[:train_size],
    _X_test=X_test[:test_size],
    _y_test=y_test[:test_size],
    _pop_size=20,
    _r_mutation=0.5,
    _p_crossover=0.7,
    _p_mutation=0.6,
    _max_iter=100,
    _min_fitness=0.95,
    _batch_size=5000,
)
g.run()

write_performance(g.evaluation_history, 'Performance.txt')

x = list(range(g.cur_iter))
plt.plot(x, g.best_acc)
plt.xlabel('iter')
plt.ylabel('best_accuracy')
plt.show()
