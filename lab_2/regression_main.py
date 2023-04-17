import random
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from neunet.neunet import NeuralNet
from preparations.ds_handle import split_dataset
from analysis.regression_error_analyser import RegressionErrorAnalyser

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
    df = df.drop(['Ro_c_кг/м3_28', 'Удельная_плотность_газа__б/р_30'], axis=1)
    df['Тна_шлейфе__С_11'] = df['Тна_шлейфе__С_11'].fillna(df['Тна_шлейфе__С_11'].mean())

    learn, test = split_dataset(df, learn_percent=80, target_columns=df.columns[-num_targets:])
    print(len(df.columns))

    hidden_1 = 120
    hidden_2 = 60
    #nn = NeuralNet(None, None, None, dir_name='./weights')
    #"""
    nn = NeuralNet(len(learn.columns) - num_targets, [hidden_1, hidden_2], num_targets, learn_rate=LEARNING_RATE,
                   start_weight_mul=START_WEIGHT_MULTIPLIER,
                   func_layers=[
                       ['me'] * (hidden_1 // 1),
                       ['me'] * (hidden_2 // 1),
                   ])
    #"""
    expected_gained_learn = nn.learn(learn, 400, categorical=False)
    learn_analysis = RegressionErrorAnalyser(expected_gained=expected_gained_learn)
    show_plots_regression(learn_analysis, learn, num_targets, "learn")

    expected_gained_test = nn.test(test, categorical=False, need_replacement=False)
    test_analysis = RegressionErrorAnalyser(expected_gained=expected_gained_test)
    show_plots_regression(test_analysis, nn.get_normalized_df(test), num_targets, "test")
    nn.save_weights('regression_weights')
    print(nn.predict(df.iloc[random.randint(0, len(df.index))][:-num_targets]))


def show_plots_regression(analyser: RegressionErrorAnalyser, df: pd.DataFrame, num_targets: int, msg: str) -> None:
    mse_values = analyser.mse()
    e_n_values = analyser.get_squared_errors()
    rmse_values = analyser.rmse()
    mae_values = analyser.mae()

    print(f'\n{msg}:\nMSE = {mse_values[-1]}\nMAE = {mae_values[-1]}\nR^2 = {analyser.r_squared(df, num_targets)}\n')

    it = np.arange(0, len(mse_values), 1)
    fig, axs = plt.subplots(1, 4)
    plt.title(msg, fontsize=15)
    axs[0].plot(it, np.array(mse_values), label='MSE')
    axs[0].grid(True)
    axs[0].set_title(label='MSE', fontsize=10)
    axs[1].plot(it, np.array(e_n_values), label='E_n')
    axs[1].grid(True)
    axs[1].set_title(label='E_n', fontsize=10)
    axs[2].plot(it, np.array(rmse_values), label='RMSE')
    axs[2].grid(True)
    axs[2].set_title(label='RMSE', fontsize=10)
    axs[3].plot(it, np.array(mae_values), label='MAE')
    axs[3].grid(True)
    axs[3].set_title(label='MAE', fontsize=10)
    plt.show()


if __name__ == '__main__':
    main()
