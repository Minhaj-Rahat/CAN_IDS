import pandas as pd

import utils_HU
import utils_HU as utHU

dosID = '0316'  # problem with dataset this can_id is showing as column name, we will add it manually later
dosFlag = 'R'
s = {'dosID': '0316', 'dosID': 'R', 'fuzzyID': '0545', 'fuzzyFlag': 'R', 'gearID': '0140', 'gearFlag': 'R',
     'rpmID': '0316',
     'rpmFlag': 'R'}


def pre_process(data_file, bits, attackType):
    data = utHU.load_data(data_file)

    # change column name, as there is no column name
    data = utils_HU.rename_columns(data, s[attackType+'ID'], 'can_ID')
    data = utils_HU.rename_columns(data, s[attackType+'Flag'], 'flag')

    # drop unnecessary column
    df = pd.concat((data['can_ID'], data['flag']), 1)

    # convert can_ID to binaries
    df['can_ID'] = df['can_ID'].apply(lambda x: utHU.con_to_bin(x, bits))  # 11/29 bits binaries
    return df

# df = pre_process('DatasetHCRL_gids/DoS_dataset.csv')
# print(df.head())
