import numpy as np
import math

import pandas as pd


def main():
    x = []
    sin_x = []
    counter = -2 * math.pi
    while counter <= 2 * math.pi:
        x.append(counter)
        sin_x.append(math.sin(counter))
        counter += 0.1

    df = pd.DataFrame(data={
        'x': x,
        'sin_x': sin_x
    })

    df.to_csv('sin_x', sep=',', index=False)


if __name__ == '__main__':
    main()
