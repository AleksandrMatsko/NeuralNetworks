import numpy as np
import neunet.funcs as nf
import typing as t


class Neuron:
    input: float
    output: float
    activation_func: t.Callable[[float], float]

    def __init__(self, activation_func: t.Callable[[float], float] = None):
        self.input = np.nan
        self.output = np.nan
        if activation_func is None:
            self.activation_func = nf.me
        else:
            self.activation_func = activation_func

    def activate(self):
        if not np.isnan(self.input):
            self.output = self.activation_func(self.input)
        return self.output

    def derivative_from_input(self, precision: float):
        if not np.isnan(self.input):
            return nf.derivative(self.activation_func, self.input, precision)
        return np.nan
