import numpy as np
import pandas as pd
import sys

from matplotlib import pyplot as plt

from neunet.neunet import NeuralNet
from neunet.layer import *
from analysis.classification_error_analyser import ClassificationErrorAnalyser


LEARNING_RATE = 0.01


def prepare_df(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df[columns] = df[columns].apply(lambda s: s.apply(lambda x: x / 255))
    return df


def convert_target_to_vector(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    target = df[column_name]
    new_target = np.zeros((len(df.index), 10))
    for label, val in target.items():
        new_target[label, int(val)] = 1.0
    new_target_df = pd.DataFrame(new_target, columns=["is_0", "is_1", "is_2", "is_3", "is_4", "is_5", "is_6", "is_7",
                                                      "is_8", "is_9"])
    df = df.drop([column_name], axis=1)
    return df.join(new_target_df)


def main():
    params = sys.argv
    if len(params) != 4:
        print('not enough arguments')
        return
    dataset_train_path = params[1]
    dataset_test_path = params[2]
    num_targets = int(params[3])
    learn = pd.read_csv(dataset_train_path)

    learn[learn.columns] = learn[learn.columns].astype(float)
    columns_list = list(learn.columns[1:])
    columns_list.append(learn.columns[0])
    learn = learn[columns_list]
    learn = prepare_df(learn, columns_list[:-1])
    learn = convert_target_to_vector(learn, learn.columns[-1])

    test = pd.read_csv(dataset_test_path)
    test = test[columns_list]
    test = prepare_df(test, columns_list[:-1])
    test = convert_target_to_vector(test, test.columns[-1])

    #print(learn)

    hidden_1 = 120
    hidden_2 = 84
    #nn = NeuralNet(dir_name='./classification_weights')
    #"""
    nn = NeuralNet(learn_rate=0.0005,  layers=[
        ConvLayer((1, 28, 28), dims_filter=(1, 3, 3), num_filters=6, start_weight_multiplier=0.1,
                  funcs=["ReLu"] * 2304, deviation=0.0),
        PoolLayer("avg", (6, 26, 26)),
        ConvLayer((6, 13, 13), dims_filter=(6, 4, 4), num_filters=16, start_weight_multiplier=0.1,
                  funcs=["ReLu"] * 576, deviation=0.0),
        PoolLayer("avg", (16, 10, 10)),
        DenseLayer(dims=(hidden_1, 400), funcs=["sigmoid"] * hidden_1, deviation=0.5),
        DenseLayer(dims=(hidden_2, hidden_1), funcs=["sigmoid"] * hidden_2, deviation=0.5),
        DenseLayer(dims=(10, hidden_2), funcs=["sigmoid"], deviation=0.5)
    ])
    expected_gained_learn = nn.learn(learn.head(10), num_epochs=1, categorical=True, need_preparations=False)
    learn_analyser = ClassificationErrorAnalyser(expected_gained_learn, 10)
    show_plots_classification(learn_analyser, "learn")
    #"""


    #expected_gained_test = nn.test(test, categorical=True, need_preparation=False)
    #test_analyser = ClassificationErrorAnalyser(expected_gained_test, 10)
    #show_plots_classification(test_analyser, "test")

    #nn.save_net("classification_weights")


def show_plots_classification(analyser: ClassificationErrorAnalyser, msg: str):
    recall = np.average(analyser.recall(), axis=1)
    precision = np.average(analyser.precision(), axis=1)
    accuracy = np.average(analyser.accuracy(), axis=1)
    f_score = np.average(analyser.f_score(), axis=1)

    border = 0.5
    ind = analyser.get_index_by_border(border)

    print(f"\n{msg}\n border = {border}\n recall = {recall[ind]}\n precision = {precision[ind]}\n "
          f"accuracy = {accuracy[ind]}\n F-score = {f_score[ind]}\n")

    tpr = analyser.tpr()
    fpr = analyser.fpr()
    fnr = analyser.fnr()

    for i in range(tpr.shape[1]):
        fig, axs = plt.subplots(1, 2)
        fig.suptitle('is_' + str(i), fontsize=15)
        axs[0].plot(fpr[:, i], fpr[:, i], fpr[:, i], tpr[:, i])
        axs[0].grid(True)
        axs[0].set_title(label='ROC', fontsize=10)
        axs[1].plot(fpr[:, i], fpr[:, i], fpr[:, i], fnr[:, i])
        axs[1].grid(True)
        axs[1].set_title(label='DET', fontsize=10)
        plt.show()

    tpr = np.average(tpr, axis=1)
    fpr = np.average(fpr, axis=1)
    fnr = np.average(fnr, axis=1)
    fig, axs = plt.subplots(1, 2)
    fig.suptitle('average', fontsize=15)
    axs[0].plot(fpr, fpr, fpr, tpr)
    axs[0].grid(True)
    axs[0].set_title(label='ROC', fontsize=10)
    axs[1].plot(fpr, fpr, fpr, fnr)
    axs[1].grid(True)
    axs[1].set_title(label='DET', fontsize=10)
    plt.show()


if __name__ == "__main__":
    main()