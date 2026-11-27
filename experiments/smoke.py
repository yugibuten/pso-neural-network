"""Quick Banknote integration check: one short PSO run on the training data."""
import numpy as np
from src.ann import ANN
from src.coupling import make_fitness
from src.data import prepare_data
from src.pso import PSO


def main():
    X, _, y, _, _ = prepare_data()
    ann = ANN([4,8,1],['tanh','logistic'],seed=0)
    best, score, history = PSO(ann.n_params,make_fitness(ann,X,y),swarm_size=20,seed=0).run(100)
    print('Training majority baseline:',np.bincount(y).max()/len(y))
    print('Initial best training accuracy:',history[0])
    print('Final best training accuracy:',score)


if __name__ == '__main__': main()
