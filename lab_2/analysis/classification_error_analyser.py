import numpy as np
import typing


class ClassificationErrorAnalyser:

    __BORDER: float

    __expected_gained: typing.List[typing.Tuple[np.array, np.array]]
    __TP: np.array
    __FP: np.array
    __FN: np.array
    __TN: np.array

    def __calc_params(self):
        if self.__expected_gained is None:
            raise ValueError("no data provided")
        tp = [0]
        fp = [0]
        fn = [0]
        tn = [0]
        for tup in self.__expected_gained:
            if tup[0] == 1:    # object belongs to class
                if tup[1] > self.__BORDER: # classifier said it is from class
                    tp.append(tp[-1] + 1)
                    fn.append(fn[-1])
                else:   # classifier says it is not from class
                    fn.append(fn[-1] + 1)
                    tp.append(tp[-1])
                fp.append(fp[-1])
                tn.append(tn[-1])
            elif tup[0] == 0:  # object doesn't belong to class
                if tup[1] > self.__BORDER: # classifier said it is from class
                    fp.append(fp[-1] + 1)
                    tn.append(tn[-1])
                else:
                    tn.append(tn[-1] + 1)
                    fp.append(fp[-1])
                tp.append(tp[-1])
                fn.append(fn[-1])
            else:
                raise ValueError("sorry analyser now works only for binary classification")
            self.__TP = np.array(tp[1:])
            self.__TN = np.array(tn[1:])
            self.__FP = np.array(fp[1:])
            self.__FN = np.array(fn[1:])

    def __init__(self, expected_gained: typing.List[typing.Tuple[np.array, np.array]], border: float = 0.5) -> None:
        self.__expected_gained = expected_gained
        self.__BORDER = border
        self.__calc_params()

    def precision(self) -> np.array:
        return np.divide(self.__TP, self.__TP + self.__FP)

    def recall(self) -> np.array:
        return np.divide(self.__TP, self.__TP + self.__FN)

    def accuracy(self) -> np.array:
        return (self.__TP + self.__TN) / len(self.__TP)

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