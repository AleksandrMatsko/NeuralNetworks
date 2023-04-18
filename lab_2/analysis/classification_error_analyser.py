import numpy as np
import typing


class ClassificationErrorAnalyser:

    __BORDERS: typing.List[float]

    __expected_gained: typing.List[typing.Tuple[np.array, np.array]]
    __TP: np.array
    __FP: np.array
    __FN: np.array
    __TN: np.array

    def __calc_for_one_border(self, border: float) -> dict:
        local_tp = 0
        local_fp = 0
        local_fn = 0
        local_tn = 0
        for tup in self.__expected_gained:
            if tup[0] == 1:  # object belongs to class
                if tup[1] >= border:  # classifier said it is from class
                    local_tp += 1
                else:  # classifier says it is not from class
                    local_fn += 1
            elif tup[0] == 0:  # object doesn't belong to class
                if tup[1] >= border:  # classifier said it is from class
                    local_fp += 1
                else:
                    local_tn += 1
            else:
                raise ValueError("sorry analyser now works only for binary classification")
        return {
            "TP": local_tp,
            "FP": local_fp,
            "TN": local_tn,
            "FN": local_fn
        }

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

    def __init__(self, expected_gained: typing.List[typing.Tuple[np.array, np.array]],
                 border_step: float = 0.05) -> None:
        self.__expected_gained = expected_gained
        self.__BORDERS = np.arange(0, 1.2, border_step).tolist()
        print(f"borders:\n {self.__BORDERS}")
        self.__calc_params()

    def precision(self) -> np.array:
        return np.divide(self.__TP, self.__TP + self.__FP)

    def recall(self) -> np.array:
        return np.divide(self.__TP, self.__TP + self.__FN)

    def accuracy(self) -> np.array:
        return (self.__TP + self.__TN) / len(self.__expected_gained)

    def tpr(self) -> np.array:
        return self.recall()

    def fpr(self) -> np.array:
        return np.divide(self.__FP, self.__FP + self.__TN)

    def fnr(self) -> np.array:
        return np.divide(self.__FN, self.__FN + self.__TP)

    def f_score(self) -> np.array:
        precision = self.precision()
        recall = self.recall()
        return 2 * np.divide(np.multiply(precision, recall), precision + recall)

    def get_index_by_border(self, border):
        return self.__BORDERS.index(border)
