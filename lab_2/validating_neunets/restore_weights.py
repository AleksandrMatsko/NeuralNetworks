import os
import numpy as np


def main():
    dir_name = '../weights'
    dir_content = os.listdir(dir_name)
    dir_content.sort()
    print(dir_content)
    weights_list = []
    for fname in dir_content:
        file_path = os.path.join(dir_name, fname)
        if os.path.isfile(file_path) and fname.endswith('.csv'):
            weights_list.append(np.loadtxt(file_path, delimiter=','))
    print(weights_list)
    for matrix in weights_list:
        print(np.shape(matrix))


if __name__ == '__main__':
    main()
