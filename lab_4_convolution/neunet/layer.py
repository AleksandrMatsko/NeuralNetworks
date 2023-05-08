import math
import typing as ty
from multiprocessing.pool import ThreadPool

import numpy as np

from neunet.neuron import Neuron


class DenseLayer:
    __NAME = "dense"
    __NUM_NEURONS: int
    __START_WEIGHT_MULTIPLIER: float

    __weight_matrix: np.array
    __dims: ty.Tuple
    __input: np.array
    __output: np.array
    __neurons: ty.List[Neuron]

    def __init__(self, start_weight_multiplier: float = 1, dims: ty.Tuple = None, weight_matrix: np.array = None,
                 funcs: ty.List[str] = None, deviation: float = 0.5):
        self.__START_WEIGHT_MULTIPLIER = start_weight_multiplier
        self.__input = None
        self.__output = None
        if weight_matrix is None and dims is not None:
            self.__weight_matrix = self.__START_WEIGHT_MULTIPLIER * (np.random.random_sample(dims) - deviation)
            self.__dims = dims
        elif weight_matrix is not None:
            self.__weight_matrix = weight_matrix
            self.__dims = np.shape(weight_matrix)
        else:
            raise ValueError("not enough args to create weight_matrix")
        self.__NUM_NEURONS = self.__dims[0]
        self.__neurons = []
        for i in range(self.__NUM_NEURONS):
            if funcs is not None and i < len(funcs):
                self.__neurons.append(Neuron(funcs[i]))
            else:
                self.__neurons.append(Neuron())


    def set_input(self, input_vec: np.array) -> None:
        #print(f"dense set_input: shape = {input_vec.shape}")
        self.__input = input_vec.reshape(self.__dims[1], 1)
        #print(f"dense set_input after reshape: shape = {self.__input.shape}")

    def get_weight_matrix(self) -> np.array:
        return self.__weight_matrix

    def get_dims(self):
        return self.__dims

    def activate(self) -> None:
        if self.__input is None:
            raise ValueError("input is not set")
        after_weights = self.__weight_matrix.dot(self.__input)
        after_weights = after_weights.reshape(len(after_weights))

        out = []
        for i in range(len(after_weights)):
            self.__neurons[i].set_input(after_weights[i])
            self.__neurons[i].activate()
            out.append(self.__neurons[i].get_output())
        self.__output = np.array(out)

    def get_output(self) -> np.array:
        if self.__output is None:
            raise ValueError("output is None")
        return self.__output

    def get_act_funcs(self) -> ty.List[str]:
        return [n.get_act_func() for n in self.__neurons]

    def calc_derivatives(self, precision: float) -> np.array:
        return np.array([n.derivative_from_input(precision) for n in self.__neurons])

    def prepare_for_deltas(self, deltas: np.array, learning_rate: float) -> ty.Tuple[np.array, bool]:
        weight_correction = np.nan_to_num(self.__input.reshape(len(self.__input), 1).dot(deltas).transpose()) \
                            * learning_rate
        tmp = deltas.dot(self.__weight_matrix)
        self.__weight_matrix = self.__weight_matrix + weight_correction
        return tmp, True

    def calc_output_delta(self, error: np.array, precision: float) -> np.array:
        return np.multiply(error, self.calc_derivatives(precision))

    def get_name(self) -> str:
        return self.__NAME


class ConvLayer:
    __DERIVATIVE_PRECISION = 0.0001
    __NAME = "conv"
    __dims_input: ty.Tuple

    __filters: ty.List[np.array]
    __input: np.array
    __output: np.array
    __derivatives: np.array

    __neurons: ty.List[Neuron]
    __dims_filter: ty.Tuple

    def __init__(self, dims_input: ty.Tuple, start_weight_multiplier: float = 1, dims_filter: ty.Tuple = None,
                 num_filters: int = 1, filters: ty.List[np.array] = None, funcs: ty.List[str] = None,
                 deviation: float = 0.5):
        self.__START_WEIGHT_MULTIPLIER = start_weight_multiplier
        self.__input = None
        self.__output = None
        self.__derivatives = None
        self.__filters = []
        if dims_filter is not None:
            for i in range(num_filters):
                self.__filters.append(self.__START_WEIGHT_MULTIPLIER * (np.random.random_sample(dims_filter) - deviation))
            self.__dims_filter = dims_filter
        elif filters is not None:
            # TODO check shapes
            self.__filters = filters
            self.__dims_filter = filters[0].shape
        else:
            raise ValueError("not enough args to create filters")
        self.__dims_input = dims_input
        self.__NUM_NEURONS = (self.__dims_input[0] - self.__dims_filter[0] + 1) * \
                             (self.__dims_input[1] - self.__dims_filter[1] + 1) * \
                             (self.__dims_input[2] - self.__dims_filter[2] + 1)
        self.__neurons = []
        for i in range(self.__NUM_NEURONS):
            if funcs is not None and i < len(funcs):
                self.__neurons.append(Neuron(funcs[i]))
            else:
                self.__neurons.append(Neuron("ReLu"))

    def set_input(self, inp: np.array) -> None:
        self.__input = inp.reshape(self.__dims_input)

    def get_output(self) -> np.array:
        return self.__output

    def activate(self) -> None:
        output = []
        all_derivatives = []
        for f in self.__filters:
            f_depth = f.shape[0]
            f_rows = f.shape[1]
            f_cols = f.shape[2]
            f_reshaped = f.reshape((f_rows * f_cols * f_depth))

            inp_rows = self.__dims_input[1]
            inp_cols = self.__dims_input[2]
            neu_rows = inp_rows - f_rows + 1
            neu_cols = inp_cols - f_cols + 1
            out_one = np.zeros(neu_rows * neu_cols)
            for i in range(neu_rows):
                for j in range(neu_cols):
                    out_one[i * neu_cols + j] = self.__input[:, i:i + f_rows, j:j + f_cols].reshape(
                        (f_rows * f_cols * f_depth)).dot(f_reshaped)
            out_neurons = []
            derivatives = []
            for i in range(len(self.__neurons)):
                out_neurons.append(self.__neurons[i].activate(inp=out_one[i]))
                derivatives.append(self.__neurons[i].derivative_from_input(self.__DERIVATIVE_PRECISION, inp=out_one[i]))
            output.append(np.array(out_neurons).reshape((neu_rows, neu_cols)))
            all_derivatives.append(np.array(derivatives).reshape((neu_rows, neu_cols)))

        self.__output = np.array(output)
        self.__derivatives = np.array(all_derivatives)

    def get_act_funcs(self) -> ty.List[str]:
        return [n.get_act_func() for n in self.__neurons]

    def calc_derivatives(self, precision: float) -> np.array:
        if self.__derivatives is not None:
            if self.__dims_input.count(1) == 2:
                return self.__derivatives.transpose()
            return self.__derivatives
        return np.array([n.derivative_from_input(precision) for n in self.__neurons]).reshape(self.__dims_input)

    def prepare_for_deltas(self, deltas: np.array, learning_rate: float) -> ty.Tuple[np.array, bool]:
        #print(f"deltas.shape = {deltas.shape}")
        filter_corrections = []
        d_depth = deltas.shape[0]
        d_rows = deltas.shape[1]
        d_cols = deltas.shape[2]
        #print(f"d_rows = {d_rows}, d_cols = {d_cols}")
        rows_conv = self.__dims_input[1] - d_rows
        cols_conv = self.__dims_input[2] - d_cols
        #print(f"rows_conv = {rows_conv}, cols_conv = {cols_conv}")
        for depth in range(d_depth):
            filter_correction = np.zeros(self.__filters[depth].shape)
            #print(f"filter[{depth}].shape = {self.__filters[depth].shape}")
            delta_reshaped = deltas[depth].reshape(d_rows * d_cols)
            for d in range(self.__dims_input[0]):
                for i in range(rows_conv - 1):
                    for j in range(cols_conv - 1):
                        #print(f"{d} {i} {j}")
                        filter_correction[d, i, j] = self.__input[d, i:i + d_rows, j:j + d_cols].reshape(
                            d_rows * d_cols).dot(delta_reshaped)
            filter_corrections.append(filter_correction * learning_rate)

        tmp = np.zeros(self.__dims_input)
        for d in range(d_depth):
            f_rows = self.__filters[d].shape[1]
            f_cols = self.__filters[d].shape[2]
            for i in range(d_rows - 1):
                for j in range(d_cols - 1):
                    tmp[:, i:i + f_rows, j:j + f_cols] = tmp[:, i:i + f_rows, j:j + f_cols] + \
                                                         self.__filters[d] * deltas[d, i, j]

        for i in range(len(filter_corrections)):
            self.__filters[i] = self.__filters[i] + filter_corrections[i]
        return tmp, True

    def get_name(self) -> str:
        return self.__NAME

    def get_filters(self) -> ty.List[np.array]:
        return self.__filters

    def get_input_dims(self) -> ty.Tuple:
        return self.__dims_input


class PoolLayer:
    __NAME = "pool"
    __pool_type: str
    __dims_input: ty.Tuple
    __input: np.array
    __output: np.array

    __POOL_ACTIVATE_DICT: dict
    __POOL_DELTAS_DICT: dict

    # for calculating deltas for max pooling
    __max_pool_indexes: ty.List[ty.Tuple]

    def __init__(self, pool_type: str, dims_input):
        self.__POOL_ACTIVATE_DICT = {
            "max": self.__activate_max,
            "avg": self.__activate_avg
        }
        self.__POOL_DELTAS_DICT = {
            "max": self.__deltas_max,
            "avg": self.__deltas_avg
        }
        if pool_type not in self.__POOL_ACTIVATE_DICT.keys():
            raise KeyError(f"no method for pooling with name {pool_type}")
        self.__pool_type = pool_type
        self.__dims_input = dims_input

    def activate(self):
        return self.__POOL_ACTIVATE_DICT.get(self.__pool_type)()

    def __activate_max(self):
        inp_rows = self.__dims_input[1]
        inp_cols = self.__dims_input[2]
        self.__output = np.zeros((self.__dims_input[0], inp_rows // 2, inp_cols // 2))
        self.__max_pool_indexes = []
        for d in range(self.__dims_input[0]):
            for i in range(0, inp_rows, 2):
                for j in range(0, inp_cols, 2):
                    ind_max = np.argmax(self.__input[d, i:i + 2, j:j + 2].reshape(4))
                    coords_max = (d, i + ind_max // 2, j + ind_max % 2)
                    self.__max_pool_indexes.append(coords_max)
                    self.__output[d, i // 2, j // 2] = self.__input[coords_max]

    def __activate_avg(self):
        inp_rows = self.__dims_input[1]
        inp_cols = self.__dims_input[2]
        self.__output = np.zeros((self.__dims_input[0], inp_rows // 2, inp_cols // 2))
        for d in range(self.__dims_input[0]):
            for i in range(0, inp_rows, 2):
                for j in range(0, inp_cols, 2):
                    self.__output[d, i // 2, j // 2] = np.average(self.__input[d, i:i+2, j:j+2].reshape(4))

    def set_input(self, inp: np.array):
        self.__input = inp.reshape(self.__dims_input)

    def prepare_for_deltas(self, deltas: np.array, learning_rate: float) -> ty.Tuple[np.array, bool]:
        return self.__POOL_DELTAS_DICT.get(self.__pool_type)(deltas), True

    def __deltas_max(self, deltas: np.array) -> np.array:
        new_deltas = np.zeros(self.__dims_input)
        delta_reshaped = deltas.reshape(len(self.__max_pool_indexes))
        for i in range(len(self.__max_pool_indexes)):
            new_deltas[self.__max_pool_indexes[i]] = delta_reshaped[i]
        return new_deltas

    def __deltas_avg(self, deltas: np.array) -> np.array:
        new_deltas = np.zeros(self.__dims_input)
        inp_depth = self.__dims_input[0]
        inp_rows = self.__dims_input[1]
        inp_cols = self.__dims_input[2]
        for d in range(inp_depth):
            for i in range(0, inp_rows, 2):
                for j in range(0, inp_cols, 2):
                    new_deltas[d, i:i + 2, j:j + 2] = np.array([deltas[d, i // 2, j // 2] / 4] * 4).reshape((2, 2))
        return new_deltas

    def calc_derivatives(self, precision: float):
        return np.ones(self.__output.shape)

    def get_output(self) -> np.array:
        return self.__output

    def get_name(self) -> str:
        return self.__NAME

    def get_pool_type(self) -> str:
        return self.__pool_type

    def get_input_dims(self) -> ty.Tuple:
        return self.__dims_input
