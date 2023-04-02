import random
import sys
import pandas as pd
from neunet.neunet import NeuralNet
from ds_handle import split_dataset

LEARNING_RATE = 0.001
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

    learn, test = split_dataset(df, learn_percent=80, target_columns=df.columns[-num_targets:])
    print(len(df.columns))

    hidden_1 = 100
    hidden_2 = 60
    nn = NeuralNet(None, None, None, dir_name='./weights')
    """
    nn = NeuralNet(len(learn.columns) - num_targets, [hidden_1, hidden_2], num_targets, learn_rate=LEARNING_RATE,
                   start_weight_mul=START_WEIGHT_MULTIPLIER,
                   func_layers=[
                       ['me'] * (hidden_1 // 1),
                       ['me'] * (hidden_2 // 1),
                   ])
    """
    #nn.learn(learn, 2000)
    nn.test(test)
    nn.save_weights('weights')
    print(nn.predict(df.iloc[random.randint(0, len(df.index))][:-num_targets]))


if __name__ == '__main__':
    main()
