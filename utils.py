import pandas as pd
from imblearn.over_sampling import SMOTE
from PyAstronomy import pyasl
import numconv


def load_data(file_name):
    return pd.read_csv(file_name)


# def scaling_data(data, num, class_column):
# Min-Max Normalization
# df = data.drop(class_column, axis=1)
# df_norm = (df - df.min()) / (df.max() - df.min())
# df_norm = pd.concat((df_norm, data[class_column]), 1)

# return df_norm

# apply smote on dataframe
def smoting(data, class_name):
    # for reproducibility purposes
    seed = 200
    # SMOTE number of neighbors
    k = 1
    x = data.loc[:, data.columns != class_name]
    y = data[class_name]

    sm = SMOTE(sampling_strategy='auto', k_neighbors=k, random_state=seed)
    X_res, y_res = sm.fit_resample(x, y)

    df = pd.concat([pd.DataFrame(X_res), pd.DataFrame(y_res)], axis=1)
    return df


# dropping columns that we do not need
def drop_data_columns(data, *cols):
    col_list = []
    for i in cols:
        col_list += [i]

    df = data.drop(col_list, axis=1)
    return df


# check whether DLC is 8 bytes

def check_dlc(x):
    diff = 8 - len(x)

    if diff != 0:
        return x + '0' * diff
    else:
        return x


## mehedi et al.
# data cleaning
def rosner_test(x, maxOutliers, alpha=0.05):
    r = pyasl.generalizedESD(x, maxOutliers, alpha, fullOutput=True)
    return r[1]

#def str_int(x):

# data integration


# data transformation
