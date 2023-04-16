import math

import numpy as np
import typing


class RegressionErrorAnalyser:

    __errors: typing.List[np.array]
    __squared_errors: typing.List[float]
    __abs_errors: typing.List[float]

    def __calc_errors(self):
        if self.__errors is not None:
            self.__squared_errors = []
            for err in self.__errors:
                val = np.nansum(np.multiply(err, err))
                self.__squared_errors.append(val)
                self.__abs_errors.append(math.sqrt(val))

    def __init__(self, errors: typing.List[np.array] = None):
        self.__errors = errors
        self.__calc_errors()

    def set_errors(self, errors) -> None:
        self.__errors = errors
        self.__calc_errors()

    def get_squared_errors(self) -> typing.List[float]:
        return self.__squared_errors

    def mse(self) -> typing.List[float]:
        mse_vals = []
        for i in range(len(self.__squared_errors)):
            mse_vals.append(sum(self.__squared_errors[0:i + 1]))
        for i in range(len(self.__squared_errors)):
            mse_vals[i] /= i + 1
        return mse_vals

    def rmse(self, mse_vals: typing.List[float] = None) -> typing.List[float]:
        if mse_vals is not None:
            return [math.sqrt(e) for e in mse_vals]
        return [math.sqrt(e) for e in self.mse()]

    def mae(self) -> typing.List[float]:
        mae_vals = []
        for i in range(len(self.__squared_errors)):
            mae_vals.append(sum(self.__abs_errors[0:i + 1]))
        for i in range(len(self.__abs_errors)):
            mae_vals[i] /= i + 1
        return mae_vals
