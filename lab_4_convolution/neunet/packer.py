import os.path

from neunet.layer import *
import numpy as np


class Packer:
    __dir_name: str
    __PACK_DICT: dict

    def __init__(self, dir_name):
        self.__dir_name = dir_name
        self.__PACK_DICT = {
            "dense": self.__pack_dense,
            "conv": self.__pack_conv,
            "pool": self.__pack_pool,
        }

    def pack(self, layer, index: int) -> dict:
        pack_func = self.__PACK_DICT.get(layer.get_name())
        if pack_func is None:
            raise ValueError(f"no method to pack layer with type {type(layer)}")
        return pack_func(layer, index)

    def __pack_conv(self, layer, index: int) -> dict:
        params = {
            "input_dims": layer.get_input_dims(),
            "act_funcs": layer.get_act_funcs(),
        }
        filters = layer.get_filters()
        prepared_str = "layer_" + str(index) + "_weight_"
        filenames = []
        for i in range(len(filters)):
            save_path = os.path.join(self.__dir_name, prepared_str + str(i))
            np.save(save_path, filters[i])
            filenames.append(save_path + ".npy")
        params["filters_files"] = filenames
        return params

    def __pack_pool(self, layer, index: int) -> dict:
        params = {
            "pool_type": layer.get_pool_type(),
            "input_dims": layer.get_input_dims()
        }
        return params

    def __pack_dense(self, layer, index: int) -> dict:
        save_path = os.path.join(self.__dir_name, "layer_" + str(index) + "_weight.csv")
        np.savetxt(save_path, layer.get_weight_matrix(), delimiter=",")
        params = {
            "act_funcs": layer.get_act_funcs(),
            "weight_filename": str(save_path)
        }
        return params


class UnPacker:
    __dir_name: str
    __UNPACK_DICT: dict

    def __init__(self, dir_name):
        self.__dir_name = dir_name
        self.__UNPACK_DICT = {
            "dense": self.__unpack_dense,
            "conv": self.__unpack_conv,
            "pool": self.__unpack_pool,
        }

    def unpack(self, layer_name: str, info: dict):
        unpack_func = self.__UNPACK_DICT.get(layer_name)
        if unpack_func is None:
            raise ValueError(f"no method to pack layer name {layer_name}")
        return unpack_func(info)

    def __unpack_conv(self, info: dict) -> ConvLayer:
        filters_filenames = info.get("filters_files")
        filters = []
        for name in filters_filenames:
            filters.append(np.load(name))
        return ConvLayer(dims_input=tuple(info.get("input_dims")), filters=filters, funcs=info.get("act_funcs"))

    def __unpack_pool(self, info: dict) -> PoolLayer:
        return PoolLayer(info.get("pool_type"), dims_input=tuple(info.get("input_dims")))

    def __unpack_dense(self, info: dict) -> DenseLayer:
        weight_matrix = np.loadtxt(info.get("weight_filename"), delimiter=",")
        if len(weight_matrix.shape) == 1:
            weight_matrix = weight_matrix.reshape((1, len(weight_matrix)))
        return DenseLayer(weight_matrix=weight_matrix, funcs=info.get("act_funcs"))

