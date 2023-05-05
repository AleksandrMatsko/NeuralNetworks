import numpy as np
import typing


class ClassificationErrorAnalyser:

    __BORDERS: typing.List[float]
    __NUM_TARGETS: int

    __expected_gained: typing.List[typing.Tuple[np.array, np.array]]
    __TP: np.array
    __FP: np.array
    __FN: np.array
    __TN: np.array

    @staticmethod
    def __calc_for_pair(pair: tuple, border: float) -> dict:
        local_tp = 0
        local_fp = 0
        local_fn = 0
        local_tn = 0
        if pair[0] == 1:  # object belongs to class
            if pair[1] >= border:  # classifier said it is from class
                local_tp += 1
            else:  # classifier says it is not from class
                local_fn += 1
        elif pair[0] == 0:  # object doesn't belong to class
            if pair[1] >= border:  # classifier said it is from class
                local_fp += 1
            else:
                local_tn += 1
        return {
            "TP": local_tp,
            "FP": local_fp,
            "TN": local_tn,
            "FN": local_fn
        }

    def __calc_for_one_border(self, border: float) -> dict:
        if self.__NUM_TARGETS == 1:
            vals = {
                "TP": 0,
                "FP": 0,
                "TN": 0,
                "FN": 0,
            }
            for tup in self.__expected_gained:
                d = self.__calc_for_pair(tup, border)
                vals["TP"] += d.get("TP")
                vals["FP"] += d.get("FP")
                vals["TN"] += d.get("TN")
                vals["FN"] += d.get("FN")
            return {
                "TP": [vals.get("TP")],
                "FP": [vals.get("FP")],
                "TN": [vals.get("TN")],
                "FN": [vals.get("FN")],
            }

        vals = {
            "TP": [],
            "FP": [],
            "TN": [],
            "FN": [],
        }
        for i in range(self.__NUM_TARGETS):
            local_vals = {
                "TP": 0,
                "FP": 0,
                "TN": 0,
                "FN": 0,
            }
            for tup in self.__expected_gained:
                d = self.__calc_for_pair((tup[0][i], tup[1][i]), border)
                local_vals["TP"] += d.get("TP")
                local_vals["FP"] += d.get("FP")
                local_vals["TN"] += d.get("TN")
                local_vals["FN"] += d.get("FN")
            vals["TP"].append(local_vals.get("TP"))
            vals["FP"].append(local_vals.get("FP"))
            vals["TN"].append(local_vals.get("TN"))
            vals["FN"].append(local_vals.get("FN"))
        return vals

    def __calc_params(self):
        if self.__expected_gained is None:
            raise ValueError("no data provided")
        tp = []
        fp = []
        fn = []
        tn = []
        for border in self.__BORDERS:
            res = self.__calc_for_one_border(border)
            tp.append(res.get("TP"))
            tn.append(res.get("TN"))
            fp.append(res.get("FP"))
            fn.append(res.get("FN"))

        self.__TP = np.array(tp)
        self.__TN = np.array(tn)
        self.__FP = np.array(fp)
        self.__FN = np.array(fn)

    def __init__(self, expected_gained: typing.List[typing.Tuple[np.array, np.array]], num_targets: int,
                 border_step: float = 0.05) -> None:
        if num_targets < 1:
            raise ValueError("num targets can't be less then zero")
        self.__NUM_TARGETS = num_targets
        self.__expected_gained = expected_gained
        self.__BORDERS = np.arange(0, 1.2, border_step).tolist()
        self.__calc_params()

    def precision(self) -> np.array:
        num_rows = len(self.__BORDERS)
        num_cols = self.__NUM_TARGETS
        total = num_cols * num_rows
        precisions = np.divide(self.__TP.reshape(total), self.__TP.reshape(total) + self.__FP.reshape(total))
        return np.average(precisions.reshape((num_rows, num_cols)), axis=1)

    def recall(self) -> np.array:
        num_rows = len(self.__BORDERS)
        num_cols = self.__NUM_TARGETS
        total = num_cols * num_rows
        recalls = np.divide(self.__TP.reshape(total), self.__TP.reshape(total) + self.__FN.reshape(total))
        return np.average(recalls.reshape((num_rows, num_cols)), axis=1)

    def accuracy(self) -> np.array:
        return np.average((self.__TP + self.__TN) / len(self.__expected_gained), axis=1)

    def tpr(self) -> np.array:
        return self.recall()

    def fpr(self) -> np.array:
        num_rows = len(self.__BORDERS)
        num_cols = self.__NUM_TARGETS
        total = num_cols * num_rows
        fprs = np.divide(self.__FP.reshape(total), self.__FP.reshape(total) + self.__TN.reshape(total))
        return np.average(fprs.reshape((num_rows, num_cols)), axis=1)

    def fnr(self) -> np.array:
        num_rows = len(self.__BORDERS)
        num_cols = self.__NUM_TARGETS
        total = num_cols * num_rows
        fnrs = np.divide(self.__FN.reshape(total), self.__FN.reshape(total) + self.__FP.reshape(total))
        return np.average(fnrs.reshape((num_rows, num_cols)), axis=1)

    def f_score(self) -> np.array:
        precision = self.precision()
        recall = self.recall()
        return 2 * np.divide(np.multiply(precision, recall), precision + recall)

    def get_index_by_border(self, border):
        return self.__BORDERS.index(border)
