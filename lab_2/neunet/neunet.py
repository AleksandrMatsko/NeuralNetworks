import json
import os.path
import time
import typing as ty
import numpy as np
import pandas as pd

from neunet.layer import Layer
import preparations.converter as cnvt


class NeuralNet:

    __NUM_TARGETS: int
    __DERIVATIVE_PRECISION: float
    __START_WEIGHT_MULTIPLIER: float
    __learning_rate: float

    __layers: ty.List[Layer]
    __weight_matrices: ty.List[np.array]
    __dims: ty.List[ty.Tuple]

    # for normalization
    __mean: ty.Optional[pd.Series]
    __std: ty.Optional[pd.Series]

    # for replacement
    __characters_to_nums: ty.Optional[dict]
    __nums_to_characters: ty.Optional[dict]

    def __init_constants(self, learn_rate: float, num_targets: int, precision: float, start_weight_mul: float):
        self.__weight_matrices = []
        self.__learning_rate = learn_rate
        self.__NUM_TARGETS = num_targets
        self.__mean = None
        self.__std = None
        self.__DERIVATIVE_PRECISION = precision
        self.__START_WEIGHT_MULTIPLIER = start_weight_mul

    def __unpack_from_dir(self, dir_name: str):
        if not os.path.exists(dir_name):
            raise FileNotFoundError(f"directory {dir_name} not found")
        dir_content = os.listdir(dir_name)
        dir_content.sort()
        self.__weight_matrices = []
        for file_name in dir_content:
            file_path = os.path.join(dir_name, file_name)
            if os.path.isfile(file_path) and file_name.endswith('.csv'):
                self.__weight_matrices.append(np.loadtxt(file_path, delimiter=','))
        self.__dims = []
        for i in range(len(self.__weight_matrices)):
            shape = np.shape(self.__weight_matrices[i])
            if len(shape) != 2:
                self.__weight_matrices[i] = self.__weight_matrices[i].reshape(1, shape[0])
            self.__dims.append(np.shape(self.__weight_matrices[i]))
        fp = open(os.path.join(dir_name, 'params.txt'), 'r')
        params = json.load(fp)
        self.__std = pd.Series(params['std'])
        self.__mean = pd.Series(params['mean'])
        self.__layers = []
        for func_list in params['act_funcs']:
            self.__layers.append(Layer(len(func_list), func_list))
        self.__DERIVATIVE_PRECISION = params['precision']
        self.__NUM_TARGETS = len(self.__layers[-1])
        self.__characters_to_nums = params["chrs_to_nums"]
        self.__nums_to_characters = params["nums_to_chrs"]
        print(self.__dims)

    def __init__(self, in_num, hidden_num: ty.List[int], out_num: int, dir_name: str = None, learn_rate: float = 0.001,
                 start_weight_mul: float = 1,
                 func_layers: ty.List[ty.List[str]] = None,
                 precision: float = 0.0001):
        if dir_name is not None:
            self.__unpack_from_dir(dir_name)
            self.__learning_rate = learn_rate
            self.__START_WEIGHT_MULTIPLIER = start_weight_mul
            return

        self.__layers = [Layer(in_num)]
        self.__dims = [(hidden_num[0], in_num)]
        for i in range(len(hidden_num)):
            if i < len(func_layers):
                self.__layers.append(Layer(hidden_num[i], funcs=func_layers[i]))
            else:
                self.__layers.append(Layer(hidden_num[i]))

            if i < len(hidden_num) - 1:
                self.__dims.append((hidden_num[i + 1], hidden_num[i]))
        self.__layers.append(Layer(out_num))
        self.__dims.append((out_num, hidden_num[-1]))
        self.__init_constants(learn_rate=learn_rate, num_targets=out_num, start_weight_mul=start_weight_mul,
                              precision=precision)
        print(self.__dims)

    def __prepare_normalization(self, df: pd.DataFrame):
        self.__mean = df.mean()
        self.__std = df.std()

    def __normalize(self, df: ty.Union[pd.DataFrame, pd.Series]) -> ty.Union[pd.DataFrame, pd.Series]:
        if self.__mean is None or self.__std is None:
            self.__prepare_normalization(df)
        return (df - self.__mean) / self.__std

    def __denormalize(self, df: ty.Union[pd.DataFrame, pd.Series]) -> ty.Union[pd.DataFrame, pd.Series]:
        if self.__mean is None or self.__std is None:
            return df
        return (df * self.__std) + self.__mean

    def __fill_random_weight_matrices(self):
        for i in range(len(self.__dims)):
            self.__weight_matrices.append(self.__START_WEIGHT_MULTIPLIER * (np.random.random_sample(self.__dims[i]) - 0.5))
            print(f'weight_matrices[{i}]:\n{self.__weight_matrices[i]}\n--------------------------------------')

    def save_weights(self, dir_name: str):
        print('saving weights...')
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        for i in range(len(self.__weight_matrices)):
            np.savetxt(os.path.join(dir_name, 'weight_' + str(i) + '.csv'), self.__weight_matrices[i], delimiter=',')
        params = {
            "mean": self.__mean.to_dict() if self.__mean is not None else None,
            "std": self.__std.to_dict() if self.__std is not None else None,
            "precision": self.__DERIVATIVE_PRECISION,
            "act_funcs": [layer.get_act_funcs() for layer in self.__layers],
            "chrs_to_nums": self.__characters_to_nums if self.__characters_to_nums is not None else None,
            "nums_to_chrs": self.__nums_to_characters if self.__nums_to_characters is not None else None
        }
        open(os.path.join(dir_name, 'params.txt'), 'w').write(json.dumps(params, sort_keys=False, indent=3))

    def __forward(self, params: np.array) -> np.array:
        vector = params.copy()
        self.__layers[0].set_inputs(vector)
        self.__layers[0].activate()
        vector = self.__layers[0].get_outputs()
        for i in range(len(self.__weight_matrices)):
            self.__layers[i + 1].set_inputs(self.__weight_matrices[i].dot(vector))
            self.__layers[i + 1].activate()
            vector = self.__layers[i + 1].get_outputs()
        return vector

    def learn(self, df: pd.DataFrame, num_epochs: int, categorical: bool, chars_to_nums: dict = None,
              nums_to_chars: dict = None,  retrain: bool = False):
        if not retrain:
            self.__fill_random_weight_matrices()
        if categorical and chars_to_nums is None and nums_to_chars is None:
            df, self.__characters_to_nums, self.__nums_to_characters = cnvt.convert_characters_to_nums(df, df.columns)
            df[df.columns] = df[df.columns].astype(float)
        elif not categorical:
            df = self.__normalize(df)
        elif categorical and chars_to_nums is not None and nums_to_chars is not None:
            self.__characters_to_nums = chars_to_nums
            self.__nums_to_characters = nums_to_chars
        expected_gained = []
        start_time = time.time()
        for epoch in range(num_epochs):
            if epoch != 0 and epoch % 100 == 0:
                print(f'epoch {epoch} error = {expected_gained[-1][0] - expected_gained[-1][1]}')
            df = df.sample(n=len(df.index)).reset_index(drop=True)
            for label, row in df.iterrows():
                params = row[:-self.__NUM_TARGETS]

                # forward
                vector = params.to_numpy()
                vector = self.__forward(vector)

                # calc errors
                expected = row[-self.__NUM_TARGETS:].to_numpy()
                expected_gained.append((expected, vector))
                error = expected - vector

                # calc local grad for output layer
                deltas = self.__layers[-1].output_delta_j(error, self.__DERIVATIVE_PRECISION)
                deltas = deltas.reshape(1, self.__NUM_TARGETS)

                # backpropagation cycle
                for i in range(len(self.__weight_matrices) - 1, -1, -1):
                    vector = self.__layers[i].get_outputs()
                    vector = vector.reshape(len(vector), 1)
                    weight_correction = np.nan_to_num(vector.dot(deltas).transpose()) * self.__learning_rate
                    if i != 0:
                        deltas = np.multiply(self.__layers[i].derivatives(self.__DERIVATIVE_PRECISION),
                                             deltas.dot(self.__weight_matrices[i]))
                    self.__weight_matrices[i] = self.__weight_matrices[i] + weight_correction

        print(f'learn time: {time.time() - start_time}')
        return expected_gained

    def test(self, df: pd.DataFrame, categorical: bool, need_replacement: bool):
        if categorical and need_replacement:
            df, self.__characters_to_nums, self.__nums_to_characters = cnvt.convert_characters_to_nums(df, df.columns)
            df[df.columns] = df[df.columns].astype(float)
        elif not categorical:
            df = self.__normalize(df)
        expected_gained = []
        for label, row in df.iterrows():
            params = row[:-self.__NUM_TARGETS]

            vector = params.to_numpy()
            vector = self.__forward(vector)

            expected = row[-self.__NUM_TARGETS:].to_numpy()
            expected_gained.append((expected, vector))
        return expected_gained

    def predict(self, params: pd.Series):
        params = (params - self.__mean[:len(params)]) / self.__std[:len(params)]
        res = self.__forward(params)
        return res * self.__std[-self.__NUM_TARGETS:] + self.__mean[-self.__NUM_TARGETS:]

    def get_normalized_df(self, df: pd.DataFrame):
        new_df = df.copy(deep=True)
        return self.__normalize(new_df)
