import sys
import pandas as pd
from neunet.neunet import NeuralNet

LEARNING_RATE = 0.005


def main():
    params = sys.argv
    if len(params) != 3:
        print('not enough arguments')
        return
    dataset_path = params[1]
    num_targets = int(params[2])
    df = pd.read_csv(dataset_path)

    # TODO: split dataset for learning and testing
    print(len(df.columns))

    # TODO: try to use some functions like sigmoid, tanh, ReLu etc.

    NN = NeuralNet(len(df.columns) - num_targets, [len(df.columns) - num_targets - 2, len(df.columns) // 2],
                   num_targets, LEARNING_RATE)
    NN.learn(df)
    NN.save_weights('weights/weight_')


if __name__ == '__main__':
    main()
