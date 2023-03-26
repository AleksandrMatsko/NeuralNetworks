import random
import typing as ty
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neunet.layer import Layer


class NeuralNet:

    # TODO: make prediction method based on __forward

    layers: ty.List[Layer]
    weight_matrices: ty.List[np.array]
    NUM_TARGETS: int
    LEARNING_RATE: float
    dims: ty.List[ty.Tuple]

    # for normalization
    mean: ty.Optional[pd.Series]
    std: ty.Optional[pd.Series]

    def __init__(self, in_num, hidden_num: ty.List[int], out_num: int, learn_rate: float,
                 func_layers: ty.List[ty.List[ty.Callable[[float], float]]] = None):
        self.layers = [Layer(in_num)]
        self.dims = [(hidden_num[0], in_num)]
        for i in range(len(hidden_num)):
            self.layers.append(Layer(hidden_num[i]))
            if i < len(hidden_num) - 1:
                self.dims.append((hidden_num[i + 1], hidden_num[i]))
        self.layers.append(Layer(out_num))
        self.dims.append((out_num, hidden_num[-1]))
        self.weight_matrices = []
        self.LEARNING_RATE = learn_rate
        self.NUM_TARGETS = out_num
        self.mean = None
        self.std = None
        print(self.dims)

    def __prepare_normalization(self, df: pd.DataFrame):
        self.mean = df.mean()
        self.std = df.std()

    def __normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.mean is None or self.std is None:
            self.__prepare_normalization(df)
        return (df - self.mean) / self.std

    def __fill_random_weight_matrices(self):
        for i in range(len(self.dims)):
            self.weight_matrices.append(
                np.fromfunction(lambda x, y:
                                (random.random() * x + random.random() * y + random.random()) / (x + y + 1) / 50,
                                self.dims[i]))

    def save_weights(self, name: str):
        print('saving weights...')
        for i in range(len(self.weight_matrices)):
            np.savetxt(name + str(i) + '.csv', self.weight_matrices[i], delimiter=',')

    def __forward(self, params: np.array) -> np.array:
        vector = params.copy()
        self.layers[0].set_inputs(vector)
        self.layers[0].activate()
        for i in range(len(self.weight_matrices)):
            self.layers[i + 1].set_inputs(self.weight_matrices[i].dot(vector))
            self.layers[i + 1].activate()
            vector = self.layers[i + 1].get_outputs()
        return vector

    def learn(self, df: pd.DataFrame, retrain: bool = False):
        if not retrain:
            self.__fill_random_weight_matrices()
        df = self.__normalize_df(df)
        counter = 0
        MSE = 0
        E_n_values = []
        MSE_values = []
        for label, row in df.iterrows():
            params = row[:-self.NUM_TARGETS]

            # forward
            vector = params.to_numpy()
            vector = self.__forward(vector)

            # calc errors
            #print(vector)
            expected = row[-self.NUM_TARGETS:]
            error = []
            for d, y in zip(expected, vector):
                if np.isnan(d):
                    error.append(np.nan)
                else:
                    error.append(d - y)
            error = np.array(error)
            E_n = np.nansum(np.multiply(error, error))
            E_n /= 2
            MSE *= counter
            MSE += E_n
            counter += 1
            MSE /= counter
            E_n_values.append(E_n)
            MSE_values.append(MSE)

            # calc local grad for output layer
            deltas = self.layers[-1].output_delta_j(error) * self.LEARNING_RATE
            deltas = deltas.reshape(1, self.NUM_TARGETS)
            #print(f'shape deltas = {np.shape(deltas)}')

            # backpropagation cycle
            for i in range(len(self.weight_matrices) - 1, -1, -1):
                vector = self.layers[i].get_outputs()
                vector = vector.reshape(len(vector), 1)
                #print(f'shape vector {np.shape(vector)}')
                self.weight_matrices[i] = self.weight_matrices[i] - np.nan_to_num(vector.dot(deltas).transpose())
                if i == 0:
                    break
                deltas = np.multiply(self.layers[i].derivatives(0.0001),
                                     deltas.dot(self.weight_matrices[i]))
                #print(f'shape deltas = {np.shape(deltas)}')

        print(MSE_values)
        it = np.arange(0, len(df.index), 1)
        plt.plot(it, np.array(MSE_values), label=r'$MSE$')
        plt.grid(True)
        plt.show()
