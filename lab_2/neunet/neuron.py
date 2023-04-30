import numpy as np
import neunet.funcs as nf
import typing as t


class Neuron:
    __input: float
    __output: float
    __activation_func: t.Callable[[float], float]
    __activation_func_name: str

    def __init__(self, activation_func: str = None):
        self.__input = np.nan
        self.__output = np.nan
        if activation_func is not None and nf.FUNC_DICT[activation_func] is not None:
            self.__activation_func = nf.FUNC_DICT[activation_func]
            self.__activation_func_name = activation_func
        else:
            self.__activation_func = nf.me
            self.__activation_func_name = 'me'

    def set_input(self, i: float):
        self.__input = i

    def get_input(self) -> float:
        return self.__input

    def get_output(self) -> float:
        return self.__output

    def activate(self) -> t.Union[float, type(np.nan)]:
        if not np.isnan(self.__input):
            self.__output = self.__activation_func(self.__input)
        return self.__output

    def derivative_from_input(self, precision: float) -> t.Union[float, type(np.nan)]:
        if not np.isnan(self.__input):
            return nf.derivative(self.__activation_func, self.__input, precision)
        return np.nan

    def get_act_func(self) -> str:
        return self.__activation_func_name
