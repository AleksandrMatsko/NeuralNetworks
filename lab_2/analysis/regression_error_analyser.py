import math

import pandas as pd
import numpy as np
import typing


class RegressionErrorAnalyser:

    __expected_gained: typing.List[typing.Tuple[np.array, np.array]]
    __errors: typing.List[np.array]

    __squared_errors: typing.List[float]
    __abs_errors: typing.List[float]
    __mse_vals: typing.List[float]

    def __init_errors(self):
        if self.__errors is not None:
            self.__squared_errors = []
            self.__abs_errors = []
            for err in self.__errors:
                val = np.nansum(np.multiply(err, err))
                val /= 2
                self.__squared_errors.append(val)
                self.__abs_errors.append(math.sqrt(val))

    def __init__(self, expected_gained: typing.List[typing.Tuple[np.array, np.array]] = None,
                 errors: typing.List[np.array] = None):
        if expected_gained is not None:
            self.__expected_gained = expected_gained
            self.__errors = []
            for tup in self.__expected_gained:
                self.__errors.append(tup[0] - tup[1])
            self.__init_errors()
            self.__mse_vals = None
            return
        if errors is not None:
            self.__errors = errors
            self.__init_errors()
            self.__mse_vals = None
            return
        raise ValueError("no parameters were provided")

    def set_errors(self, errors) -> None:
        self.__errors = errors
        self.__init_errors()

    def get_squared_errors(self) -> typing.List[float]:
        return self.__squared_errors

    def mse(self) -> typing.List[float]:
        mse_vals = []
        for i in range(len(self.__squared_errors)):
            if i == 0:
                mse_vals.append(self.__squared_errors[0])
            else:
                mse_vals.append(self.__squared_errors[i] + mse_vals[i - 1])
        for i in range(len(self.__squared_errors)):
            mse_vals[i] /= i + 1
        self.__mse_vals = mse_vals
        return mse_vals

    def rmse(self, mse_vals: typing.List[float] = None) -> typing.List[float]:
        if mse_vals is not None:
            return [math.sqrt(e) for e in mse_vals]
        if self.__mse_vals is not None:
            return [math.sqrt(e) for e in self.__mse_vals]
        return [math.sqrt(e) for e in self.mse()]

    def mae(self) -> typing.List[float]:
        mae_vals = []
        for i in range(len(self.__abs_errors)):
            if i == 0:
                mae_vals.append(self.__abs_errors[0])
            else:
                mae_vals.append(self.__abs_errors[i] + mae_vals[i - 1])
        for i in range(len(self.__abs_errors)):
            mae_vals[i] /= i + 1
        return mae_vals

    def r_squared(self, df: pd.DataFrame, num_targets: int, mse: float = None) -> float:
        if len(df.index) != len(self.__expected_gained):
            return None
        sum_deviations_expected = 0
        for pair in self.__expected_gained:
            deviation_expected = pair[0] - df.mean(skipna=True)[-num_targets:].to_numpy().reshape(num_targets, 1)
            sum_deviations_expected += np.nansum(np.multiply(deviation_expected, deviation_expected))
        tmp = len(self.__expected_gained) / sum_deviations_expected
        if self.__mse_vals is not None:
            return 1 - self.__mse_vals[-1] * 2 * tmp
        if mse is not None:
            return 1 - mse * 2 * tmp
        return 1 - self.mse()[-1] * 2 * tmp
