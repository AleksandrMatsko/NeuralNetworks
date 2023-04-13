import json
import math
import os.path
import time
import typing as ty
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neunet.layer import Layer


class NeuralNet:

    __NUM_TARGETS: int
    __DERIVATIVE_PRECISION: float
    __START_WEIGHT_MULTIPLIER: float
    __learning_rate: float

    __layers: ty.List[Layer]
    __weight_matrices: ty.List[np.array]
    __dims: ty.List[ty.Tuple]

    # for normalization
    __mean: ty.Optional[pd.Series]
    __std: ty.Optional[pd.Series]

    def __init_constants(self, learn_rate: float, num_targets: int, precision: float, start_weight_mul: float):
        self.__weight_matrices = []
        self.__learning_rate = learn_rate
        self.__NUM_TARGETS = num_targets
        self.__mean = None
        self.__std = None
        self.__DERIVATIVE_PRECISION = precision
        self.__START_WEIGHT_MULTIPLIER = start_weight_mul

    def __unpack_from_dir(self, dir_name: str):
        dir_content = os.listdir(dir_name)
        dir_content.sort()
        self.__weight_matrices = []
        for file_name in dir_content:
            file_path = os.path.join(dir_name, file_name)
            if os.path.isfile(file_path) and file_name.endswith('.csv'):
                self.__weight_matrices.append(np.loadtxt(file_path, delimiter=','))
        self.__dims = []
        for matrix in self.__weight_matrices:
            self.__dims.append(np.shape(matrix))
        fp = open(os.path.join(dir_name, 'params.txt'), 'r')
        params = json.load(fp)
        self.__std = pd.Series(params['std'])
        self.__mean = pd.Series(params['mean'])
        self.__layers = []
        for func_list in params['act_funcs']:
            self.__layers.append(Layer(len(func_list), func_list))
        self.__DERIVATIVE_PRECISION = params['precision']
        self.__NUM_TARGETS = len(self.__layers[-1])
        print(self.__dims)

    def __init__(self, in_num, hidden_num: ty.List[int], out_num: int, dir_name: str = None, learn_rate: float = 0.001,
                 start_weight_mul: float = 1,
                 func_layers: ty.List[ty.List[str]] = None,
                 precision: float = 0.0001):
        if dir_name is not None:
            self.__unpack_from_dir(dir_name)
            self.__learning_rate = learn_rate
            self.__START_WEIGHT_MULTIPLIER = start_weight_mul
            return

        self.__layers = [Layer(in_num)]
        self.__dims = [(hidden_num[0], in_num)]
        for i in range(len(hidden_num)):
            if i < len(func_layers):
                self.__layers.append(Layer(hidden_num[i], funcs=func_layers[i]))
            else:
                self.__layers.append(Layer(hidden_num[i]))

            if i < len(hidden_num) - 1:
                self.__dims.append((hidden_num[i + 1], hidden_num[i]))
        self.__layers.append(Layer(out_num))
        self.__dims.append((out_num, hidden_num[-1]))
        self.__init_constants(learn_rate=learn_rate, num_targets=out_num, start_weight_mul=start_weight_mul,
                              precision=precision)
        print(self.__dims)

    def __prepare_normalization(self, df: pd.DataFrame):
        self.__mean = df.mean()
        self.__std = df.std()

    def __normalize(self, df: ty.Union[pd.DataFrame, pd.Series]) -> ty.Union[pd.DataFrame, pd.Series]:
        if self.__mean is None or self.__std is None:
            self.__prepare_normalization(df)
        return (df - self.__mean) / self.__std

    def __denormalize(self, df: ty.Union[pd.DataFrame, pd.Series]) -> ty.Union[pd.DataFrame, pd.Series]:
        if self.__mean is None or self.__std is None:
            return df
        return (df * self.__std) + self.__mean

    def __fill_random_weight_matrices(self):
        for i in range(len(self.__dims)):
            self.__weight_matrices.append(self.__START_WEIGHT_MULTIPLIER * (np.random.random_sample(self.__dims[i]) - 0.5))
            print(f'weight_matrices[{i}]:\n{self.__weight_matrices[i]}\n--------------------------------------')

    def save_weights(self, dir_name: str):
        print('saving weights...')
        for i in range(len(self.__weight_matrices)):
            np.savetxt(os.path.join(dir_name, 'weight_' + str(i) + '.csv'), self.__weight_matrices[i], delimiter=',')
        params = {
            "mean": self.__mean.to_dict(),
            "std": self.__std.to_dict(),
            "precision": self.__DERIVATIVE_PRECISION,
            "act_funcs": [layer.get_act_funcs() for layer in self.__layers]
        }
        open(os.path.join(dir_name, 'params.txt'), 'w').write(json.dumps(params, sort_keys=False, indent=3))

    def __forward(self, params: np.array) -> np.array:
        vector = params.copy()
        self.__layers[0].set_inputs(vector)
        self.__layers[0].activate()
        vector = self.__layers[0].get_outputs()
        for i in range(len(self.__weight_matrices)):
            self.__layers[i + 1].set_inputs(self.__weight_matrices[i].dot(vector))
            self.__layers[i + 1].activate()
            vector = self.__layers[i + 1].get_outputs()
        return vector

    def __calc_e_n(self, expected, vector):
        error = []
        for d, y in zip(expected, vector):
            if np.isnan(d):
                error.append(np.nan)
            else:
                error.append(d - y)
        error = np.array(error)
        e_n = np.nansum(np.multiply(error, error))
        e_n /= 2
        return e_n, error

    def learn(self, df: pd.DataFrame, num_epochs: int, retrain: bool = False):
        if not retrain:
            self.__fill_random_weight_matrices()
        df = self.__normalize(df)
        counter = 0
        mse = 0
        mae = 0
        e_n_values = []
        mse_values = []
        mae_values = []
        start_time = time.time()
        for epoch in range(num_epochs):
            if epoch != 0 and epoch % 100 == 0:
                print(f'epoch {epoch} E_n = {e_n_values[-1]}')
            df = df.sample(n=len(df.index)).reset_index(drop=True)
            for label, row in df.iterrows():
                params = row[:-self.__NUM_TARGETS]

                # forward
                vector = params.to_numpy()
                vector = self.__forward(vector)

                # calc errors
                # print(vector)
                expected = row[-self.__NUM_TARGETS:].to_numpy()
                #print(f'expected:\n{expected}')
                e_n, error = self.__calc_e_n(expected, vector)
                #print(f'error:\n{error}')
                mse *= counter
                mae *= counter
                mse += e_n
                mae += math.sqrt(e_n)
                counter += 1
                mse /= counter
                mae /= counter

                e_n_values.append(e_n / len(error))
                mse_values.append(mse)
                mae_values.append(mae)

                # calc local grad for output layer

                deltas = self.__layers[-1].output_delta_j(error, self.__DERIVATIVE_PRECISION)
                deltas = deltas.reshape(1, self.__NUM_TARGETS)
                #print(f'deltas last layer:\n{deltas}')
                # print(f'shape deltas = {np.shape(deltas)}')

                # backpropagation cycle
                for i in range(len(self.__weight_matrices) - 1, -1, -1):
                    vector = self.__layers[i].get_outputs()
                    vector = vector.reshape(len(vector), 1)
                    # print(f'shape vector {np.shape(vector)}')
                    weight_correction = np.nan_to_num(vector.dot(deltas).transpose()) * self.__learning_rate
                    # print(f'weight_correction[{i}]:\n{weight_correction}\n===============================')
                    if i != 0:
                        deltas = np.multiply(self.__layers[i].derivatives(self.__DERIVATIVE_PRECISION),
                                             deltas.dot(self.__weight_matrices[i]))
                    #print(f'deltas layer[{i}]:\n{deltas}')
                    self.__weight_matrices[i] = self.__weight_matrices[i] + weight_correction
                    #print(f'weight_matrix[{i}]:\n{self.__weight_matrices[i]}')
                    # print(f'shape deltas = {np.shape(deltas)}')

        print(f'learn time: {time.time() - start_time}')
        it = np.arange(0, len(df.index) * num_epochs, 1)
        fig, axs = plt.subplots(1, 4)
        plt.title('learn', fontsize=15)
        axs[0].plot(it, np.array(mse_values), label='MSE')
        axs[0].grid(True)
        axs[0].set_title(label='MSE', fontsize=10)
        axs[1].plot(it, np.array(e_n_values), label='E_n')
        axs[1].grid(True)
        axs[1].set_title(label='E_n', fontsize=10)
        axs[2].plot(it, np.array([math.sqrt(e) for e in mse_values]), label='RMSE')
        axs[2].grid(True)
        axs[2].set_title(label='RMSE', fontsize=10)
        axs[3].plot(it, np.array(mae_values), label='MAE')
        axs[3].grid(True)
        axs[3].set_title(label='MAE', fontsize=10)
        plt.show()

        print(f'\nlearn\nMSE = {mse}\nMAE = {mae}\n')

    def test(self, df: pd.DataFrame):
        df = self.__normalize(df)
        counter = 0
        mse = 0
        mae = 0
        e_n_values = []
        mse_values = []
        mae_values = []
        sum_deviations_expected = 0
        for label, row in df.iterrows():
            params = row[:-self.__NUM_TARGETS]

            vector = params.to_numpy()
            vector = self.__forward(vector)

            expected = row[-self.__NUM_TARGETS:].to_numpy()
            e_n, error = self.__calc_e_n(expected, vector)
            mse *= counter
            mae *= counter
            mse += e_n
            print(f'E_n = {e_n}\n{self.__denormalize(row)}')
            mae += math.sqrt(e_n)
            deviation_expected = expected - df.mean(skipna=True)[-self.__NUM_TARGETS:].to_numpy()\
                .reshape(self.__NUM_TARGETS, 1)
            sum_deviations_expected += np.nansum(np.multiply(deviation_expected, deviation_expected))
            counter += 1
            mse /= counter
            mae /= counter
            e_n_values.append(e_n / len(error))
            mse_values.append(mse)
            mae_values.append(mae)

        print(f'\ntest:\nMSE = {mse}\nMAE = {mae}\nR^2 = {1 - mse * counter * 2 / sum_deviations_expected}\n')

        it = np.arange(0, len(df.index), 1)
        fig, axs = plt.subplots(1, 4)
        plt.title('test', fontsize=15)
        axs[0].plot(it, np.array(mse_values), label='MSE')
        axs[0].grid(True)
        axs[0].set_title(label='MSE', fontsize=10)
        axs[1].plot(it, np.array(e_n_values), label='E_n')
        axs[1].grid(True)
        axs[1].set_title(label='E_n', fontsize=10)
        axs[2].plot(it, np.array([math.sqrt(e) for e in mse_values]), label='RMSE')
        axs[2].grid(True)
        axs[2].set_title(label='RMSE', fontsize=10)
        axs[3].plot(it, np.array(mae_values), label='MAE')
        axs[3].grid(True)
        axs[3].set_title(label='MAE', fontsize=10)
        plt.show()

    def predict(self, params: pd.Series):
        params = (params - self.__mean[:len(params)]) / self.__std[:len(params)]
        res = self.__forward(params)
        return res * self.__std[-self.__NUM_TARGETS:] + self.__mean[-self.__NUM_TARGETS:]
