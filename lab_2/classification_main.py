import pandas as pd
import sys

from matplotlib import pyplot as plt

from neunet.neunet import NeuralNet
from preparations.ds_handle import split_dataset
from analysis.classification_error_analyser import ClassificationErrorAnalyser
import preparations.converter as cnvt


LEARNING_RATE = 0.0005
START_WEIGHT_MULTIPLIER = 1


def main():
    params = sys.argv
    if len(params) != 3:
        print('not enough arguments')
        return
    dataset_path = params[1]
    num_targets = int(params[2])
    df = pd.read_csv(dataset_path)

    df, chars_to_nums, nums_to_chars = cnvt.convert_characters_to_nums(df, df.columns)
    df[df.columns] = df[df.columns].astype(float)

    learn, test = split_dataset(df, learn_percent=60, target_columns=df.columns[-num_targets:])
    print(len(df.columns))
    #test = cnvt.convert_vals_with_rule(test, test.columns, nums_to_chars)

    hidden_1 = 120
    hidden_2 = 60
    #nn = NeuralNet(None, None, None, dir_name='./classification_weights')
    #"""
    nn = NeuralNet(len(df.columns) - num_targets, [hidden_1, hidden_2], num_targets, learn_rate=LEARNING_RATE,
                   start_weight_mul=START_WEIGHT_MULTIPLIER,
                   func_layers=[
                       ['sigmoid'] * (hidden_1 // 1),
                       ['sigmoid'] * (hidden_2 // 1),
                   ])
    expected_gained_learn = nn.learn(learn, num_epochs=10, categorical=True, chars_to_nums=chars_to_nums,
                                     nums_to_chars=nums_to_chars)
    learn_analyser = ClassificationErrorAnalyser(expected_gained_learn)
    show_plots_classification(learn_analyser, "learn")
    #"""

    expected_gained_test = nn.test(test, categorical=True, need_replacement=False)
    test_analyser = ClassificationErrorAnalyser(expected_gained_test)
    show_plots_classification(test_analyser, "test")

    nn.save_weights("classification_weights")


def show_plots_classification(analyser: ClassificationErrorAnalyser, msg: str):
    recall = analyser.recall()
    precision = analyser.precision()
    accuracy = analyser.accuracy()
    f_score = analyser.f_score()

    border = 0.5
    ind = analyser.get_index_by_border(border)

    print(f"\n{msg}\n border = {border}\n recall = {recall[ind]}\n precision = {precision[ind]}\n "
          f"accuracy = {accuracy[ind]}\n F-score = {f_score[ind]}\n")

    tpr = analyser.tpr()
    fpr = analyser.fpr()
    fnr = analyser.fnr()

    fig, axs = plt.subplots(1, 2)
    plt.title('test', fontsize=15)
    axs[0].plot(fpr, fpr, fpr, tpr)
    axs[0].grid(True)
    axs[0].set_title(label='ROC', fontsize=10)
    axs[1].plot(fpr, fpr, fpr, fnr)
    axs[1].grid(True)
    axs[1].set_title(label='DET', fontsize=10)
    plt.show()


if __name__ == "__main__":
    main()
