import typing as ty
import numpy as np

from neunet.neuron import Neuron


class Layer:
    neurons: ty.List[Neuron]
    deltas: np.array

    def __init__(self, num_neurons: int, funcs: list = None):
        self.neurons = []
        for i in range(num_neurons):
            if funcs is not None and i < len(funcs):
                self.neurons.append(Neuron(funcs[i]))
            else:
                self.neurons.append(Neuron())

    def __len__(self):
        return len(self.neurons)

    def get_outputs(self) -> np.array:
        return np.array([n.output for n in self.neurons])

    def set_inputs(self, inputs: np.array) -> None:
        for i in range(len(inputs)):
            self.neurons[i].input = inputs[i]

    def activate(self) -> None:
        for n in self.neurons:
            n.activate()

    def derivatives(self, precision: float) -> np.array:
        return np.array([n.derivative_from_input(precision) for n in self.neurons])

    def output_delta_j(self, error: np.array) -> np.array:
        self.deltas = np.array([error[j] * self.neurons[j].derivative_from_input(0.0001) for j in range(len(error))])
        return self.deltas
