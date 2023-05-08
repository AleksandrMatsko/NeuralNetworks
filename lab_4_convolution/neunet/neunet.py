import json
import os.path
import time
import typing as ty
import numpy as np
import pandas as pd

from neunet.layer import DenseLayer
from neunet.packer import Packer, UnPacker
import preparations.converter as cnvt


class NeuralNet:
    __NUM_TARGETS: int
    __DERIVATIVE_PRECISION: float
    __learning_rate: float

    __layers: ty.List[DenseLayer]

    # for normalization
    __mean: ty.Optional[pd.Series]
    __std: ty.Optional[pd.Series]

    # for replacement
    __characters_to_nums: ty.Optional[dict]
    __nums_to_characters: ty.Optional[dict]

    def __init_constants(self, learn_rate: float, num_targets: int, precision: float):
        self.__weight_matrices = []
        self.__learning_rate = learn_rate
        self.__NUM_TARGETS = num_targets
        self.__mean = None
        self.__std = None
        self.__characters_to_nums = None
        self.__nums_to_characters = None
        self.__DERIVATIVE_PRECISION = precision

    def __unpack_from_dir(self, dir_name: str):
        if not os.path.exists(dir_name):
            raise FileNotFoundError(f"directory {dir_name} not found")
        dir_content = os.listdir(dir_name)
        dir_content.sort()

        fp = open(os.path.join(dir_name, 'params.txt'), 'r')
        params = json.load(fp)
        self.__std = pd.Series(params['std'], dtype=float)
        self.__mean = pd.Series(params['mean'], dtype=float)
        self.__characters_to_nums = params["chrs_to_nums"]
        self.__nums_to_characters = params["nums_to_chrs"]
        self.__NUM_TARGETS = params["num_targets"]
        layers = params["layers"]
        unpacker = UnPacker(dir_name)
        self.__layers = []
        for i in range(len(layers)):
            self.__layers.append(unpacker.unpack(layers[i].get("layer_name"),
                                                 layers[i].get("layer_params")))

    def save_net(self, dir_name: str):
        print('saving weights...')
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        params = {
            "mean": self.__mean.to_dict() if self.__mean is not None else None,
            "std": self.__std.to_dict() if self.__std is not None else None,
            "chrs_to_nums": self.__characters_to_nums if self.__characters_to_nums is not None else None,
            "nums_to_chrs": self.__nums_to_characters if self.__nums_to_characters is not None else None,
            "num_targets": self.__NUM_TARGETS
        }
        packer = Packer(dir_name)
        layers = []
        for i in range(len(self.__layers)):
            layers.append({"layer_name": self.__layers[i].get_name(),
                           "layer_params": packer.pack(self.__layers[i], i)})
        params["layers"] = layers
        open(os.path.join(dir_name, 'params.txt'), 'w').write(json.dumps(params, sort_keys=False, indent=3))

    def __init__(self, dir_name: str = None, learn_rate: float = 0.001, layers: list = None, precision: float = 0.0001):
        if dir_name is not None:
            self.__unpack_from_dir(dir_name)
            self.__learning_rate = learn_rate
            return
        elif layers is not None:
            self.__layers = layers
        else:
            raise ValueError("not enough data to create neural net")

        self.__init_constants(learn_rate=learn_rate, num_targets=self.__layers[-1].get_dims()[0], precision=precision)

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

    def __forward(self, params: np.array) -> np.array:
        vector = params.copy()
        for i in range(len(self.__layers)):
            #if self.__layers[i].get_name() == "dense":
            #    print(f"{vector}\n")
            #print(f"vector_shape = {vector.shape}")
            self.__layers[i].set_input(vector)
            self.__layers[i].activate()
            vector = self.__layers[i].get_output()
        return vector

    def learn(self, df: pd.DataFrame, num_epochs: int, categorical: bool, chars_to_nums: dict = None,
              nums_to_chars: dict = None, need_preparations: bool = False):
        if categorical and need_preparations and chars_to_nums is None and nums_to_chars is None:
            df, self.__characters_to_nums, self.__nums_to_characters = cnvt.convert_characters_to_nums(df, df.columns)
            df[df.columns] = df[df.columns].astype(float)
        elif not categorical and need_preparations:
            df = self.__normalize(df)
        elif categorical and need_preparations and chars_to_nums is not None and nums_to_chars is not None:
            self.__characters_to_nums = chars_to_nums
            self.__nums_to_characters = nums_to_chars
        expected_gained = []
        counter = 0
        start_time = time.time()
        for epoch in range(num_epochs):
            if epoch != 0 and epoch % 100 == 0:
                print(f'epoch {epoch} error = {expected_gained[-1][0] - expected_gained[-1][1]}')
            #df = df.sample(n=len(df.index)).reset_index(drop=True)
            for label, row in df.iterrows():
                params = row[:-self.__NUM_TARGETS]

                # forward
                vector = params.to_numpy()
                vector = self.__forward(vector)

                # calc errors
                expected = row[-self.__NUM_TARGETS:].to_numpy()
                expected_gained.append((expected, vector))
                error = expected - vector
                if counter % 1000 == 0 and counter != 0:
                    print(f"{counter}: expected = {expected}, gained = {vector}, error = {error}")
                    print(f'learn time: {time.time() - start_time}')
                counter += 1

                # calc local grad for output layer
                deltas = self.__layers[-1].calc_output_delta(error, self.__DERIVATIVE_PRECISION)
                deltas = deltas.reshape(1, self.__NUM_TARGETS)

                # backpropagation cycle
                for i in range(len(self.__layers) - 1, -1, -1):
                    #print(f"{i}: deltas:\n{deltas}")
                    prepare_deltas, need_more_calc = self.__layers[i].prepare_for_deltas(deltas, self.__learning_rate)
                    #print(f"{i}: prepare_deltas.shape = {prepare_deltas.shape}")
                    if i != 0 and need_more_calc:
                        derivatives = self.__layers[i - 1].calc_derivatives(self.__DERIVATIVE_PRECISION)
                        #print(f"{i} shape = {derivatives.shape}\nderivatives:\n{derivatives}")
                        if len(derivatives.shape) > len(prepare_deltas.shape) and len(derivatives.shape) == 3:
                            prepare_deltas = prepare_deltas.reshape(derivatives.shape)
                        deltas = np.multiply(self.__layers[i - 1].calc_derivatives(self.__DERIVATIVE_PRECISION),
                                             prepare_deltas)

        print(f'\nfinal learn time: {time.time() - start_time}')
        return expected_gained

    def test(self, df: pd.DataFrame, categorical: bool, need_preparation: bool):
        if categorical and need_preparation:
            df, self.__characters_to_nums, self.__nums_to_characters = cnvt.convert_characters_to_nums(df, df.columns)
            df[df.columns] = df[df.columns].astype(float)
        elif not categorical and need_preparation:
            df = self.__normalize(df)
        expected_gained = []
        for label, row in df.iterrows():
            params = row[:-self.__NUM_TARGETS]

            vector = params.to_numpy()
            vector = self.__forward(vector)

            expected = row[-self.__NUM_TARGETS:].to_numpy()
            expected_gained.append((expected, vector))
        return expected_gained

    def get_normalized_df(self, df: pd.DataFrame):
        new_df = df.copy(deep=True)
        return self.__normalize(new_df)
