import typing as ty
import numpy as np

from neunet.neuron import Neuron


class Layer:
    __neurons: ty.List[Neuron]

    def __init__(self, num_neurons: int, funcs: ty.List[str] = None):
        self.__neurons = []
        for i in range(num_neurons):
            if funcs is not None and i < len(funcs):
                self.__neurons.append(Neuron(funcs[i]))
            else:
                self.__neurons.append(Neuron())

    def __len__(self) -> int:
        return len(self.__neurons)

    def get_outputs(self) -> np.array:
        return np.array([n.get_output() for n in self.__neurons])

    def set_inputs(self, inputs: np.array) -> None:
        for i in range(len(inputs)):
            self.__neurons[i].set_input(inputs[i])

    def activate(self) -> None:
        for n in self.__neurons:
            n.activate()

    def derivatives(self, precision: float) -> np.array:
        return np.array([n.derivative_from_input(precision) for n in self.__neurons])

    def output_delta_j(self, error: np.array, precision: float) -> np.array:
        return np.array([error[j] * self.__neurons[j].derivative_from_input(precision) for j in range(len(error))])

    def get_act_funcs(self) -> ty.List[str]:
        return [n.get_act_func() for n in self.__neurons]

