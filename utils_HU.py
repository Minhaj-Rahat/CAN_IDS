import pandas as pd
import numpy as np


def load_data(file_name):
    return pd.read_csv(file_name)


def rename_columns(df, old, new):
    return df.rename(columns={old: new})


def con_to_bin(x, num_bit, base=16):
    return bin(int(x, base))[2:].zfill(num_bit)


def four_data_grid_eleven(x):
    x = x[::-1]  # reverse the binaries for easier implementation
    data_grid = np.zeros((4, 4))
    count = 0
    for i in range(4):
        data_grid[0][i] = int(x[count])
        count += 1
    for i in range(1, 4):
        data_grid[i][3] = int(x[count])
        count += 1
    for i in range(2, -1, -1):
        data_grid[3][i] = int(x[count])
        count += 1
    data_grid[2][0] = int(x[count])
    return data_grid


def six_data_grid_twentynine(x):
    x = x[::-1]  # reverse the binaries for easier implementation
    data_grid = np.zeros((6, 6))
    count = 0

    for i in range(1, 5):
        for j in range(1, 5):
            data_grid[i][j] = int(x[count])
            count += 1
    for i in range(6):
        data_grid[0][i] = int(x[count])
        count += 1

    for i in range(1, 4):
        for j in [0, 5]:
            data_grid[i][j] = int(x[count])
            count += 1

    data_grid[4][0] = int(x[count])
    return data_grid

# def four_mosiac_grid():
