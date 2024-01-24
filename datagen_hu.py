import utils_HU as ut_hu
import pre_process_HU as pre_hu
import numpy as np
import pickle
def gen_data(data_files, dest_files):
    # generate dos data for 4by4 data_grid and 4by4 mosaic


    data_file = data_files[0] #"DatasetHCRL_gids/DoS_dataset.csv"

    df = pre_hu.pre_process(data_file, 11,'dos')
    loop_count = len(df) // 16
    # loop_count = 1

    # can_ids numpy array
    ids = df['can_ID'].to_numpy()
    flags = df['flag'].to_numpy()

    # arrays to hold X and Y data
    x = []
    y = []

    print("generating Mosaic Data....")
    count = 0
    for i in range(loop_count):
        dlist = []
        ylist = []
        for j in range(4):

            dat_grid = ut_hu.four_data_grid_eleven(ids[count])
            ylist.append(flags[count])
            count += 1
            for k in range(3):
                dt_grid = ut_hu.four_data_grid_eleven(ids[count])
                dat_grid = np.concatenate((dat_grid, dt_grid), axis=1)
                ylist.append(flags[count])
                count += 1
            dlist.append(dat_grid)
        d_grid = dlist[0]
        for l in range(1, 4):
            d_grid = np.concatenate((d_grid, dlist[l]), axis=0)
        d_grid = d_grid[:, :, np.newaxis]

        x.append(d_grid)
        if 'T' in ylist:
            y.append('T')
        else:
            y.append('R')

    x = np.array(x)
    y = np.array(y)

    f = open(dest_files[0], 'wb')   #'data_DOS_HU'
    pickle.dump((x, y), f)
    f.close()
    # print(ids[0:16])
    # print(f'Shape of X: {x.shape}')
    # print(y)


    # generate fuzzy data for 6by6 data_grid and 8by8 mosaic

    data_file = data_files[1] #"DatasetHCRL_gids/Fuzzy_dataset.csv"

    df = pre_hu.pre_process(data_file, 29, 'fuzzy')
    loop_count = len(df) // 64
    # loop_count = 1

    # can_ids numpy array
    ids = df['can_ID'].to_numpy()
    flags = df['flag'].to_numpy()

    # arrays to hold X and Y data
    x = []
    y = []

    print("generating Mosaic Data....")
    count = 0
    for i in range(loop_count):
        dlist = []
        ylist = []
        for j in range(8):

            dat_grid = ut_hu.six_data_grid_twentynine(ids[count])
            ylist.append(flags[count])
            count += 1
            for k in range(7):
                dt_grid = ut_hu.six_data_grid_twentynine(ids[count])
                dat_grid = np.concatenate((dat_grid, dt_grid), axis=1)
                ylist.append(flags[count])
                count += 1
            dlist.append(dat_grid)
        d_grid = dlist[0]
        for l in range(1, 8):
            d_grid = np.concatenate((d_grid, dlist[l]), axis=0)
        d_grid = d_grid[:, :, np.newaxis]

        x.append(d_grid)
        if 'T' in ylist:
            y.append('T')
        else:
            y.append('R')

    x = np.array(x)
    y = np.array(y)

    f = open(dest_files[1], 'wb') #'Dataset/Hu/data_fuzzy_29_HU'
    pickle.dump((x, y), f)
    f.close()
    # print(ids[0:16])
    print(f'Shape of X: {x.shape}')
    # print(y)


    # generate gear data for 6by6 data_grid and 6by6 mosaic

    data_file = data_files[2] #"DatasetHCRL_gids/gear_dataset.csv"

    df = pre_hu.pre_process(data_file, 29, 'gear')
    loop_count = len(df) // 36
    # loop_count = 1

    # can_ids numpy array
    ids = df['can_ID'].to_numpy()
    flags = df['flag'].to_numpy()

    # arrays to hold X and Y data
    x = []
    y = []

    print("generating Mosaic Data....")
    count = 0
    for i in range(loop_count):
        dlist = []
        ylist = []
        for j in range(6):

            dat_grid = ut_hu.six_data_grid_twentynine(ids[count])
            ylist.append(flags[count])
            count += 1
            for k in range(5):
                dt_grid = ut_hu.six_data_grid_twentynine(ids[count])
                dat_grid = np.concatenate((dat_grid, dt_grid), axis=1)
                ylist.append(flags[count])
                count += 1
            dlist.append(dat_grid)
        d_grid = dlist[0]
        for l in range(1, 6):
            d_grid = np.concatenate((d_grid, dlist[l]), axis=0)
        d_grid = d_grid[:, :, np.newaxis]

        x.append(d_grid)
        if 'T' in ylist:
            y.append('T')
        else:
            y.append('R')

    x = np.array(x)
    y = np.array(y)

    f = open(dest_files[2], 'wb') #'Dataset/Hu/data_gear_29_HU'
    pickle.dump((x, y), f)
    f.close()
    # print(ids[0:16])
    print(f'Shape of X: {x.shape}')
    # print(y)


    # generate rpm data for 6by6 data_grid and 6by6 mosaic
    data_file = data_files[3] #"DatasetHCRL_gids/RPM_dataset.csv"

    df = pre_hu.pre_process(data_file, 29, 'rpm')
    loop_count = len(df) // 36
    # loop_count = 1

    # can_ids numpy array
    ids = df['can_ID'].to_numpy()
    flags = df['flag'].to_numpy()

    # arrays to hold X and Y data
    x = []
    y = []

    print("generating Mosaic Data....")
    count = 0
    for i in range(loop_count):
        dlist = []
        ylist = []
        for j in range(6):

            dat_grid = ut_hu.six_data_grid_twentynine(ids[count])
            ylist.append(flags[count])
            count += 1
            for k in range(5):
                dt_grid = ut_hu.six_data_grid_twentynine(ids[count])
                dat_grid = np.concatenate((dat_grid, dt_grid), axis=1)
                ylist.append(flags[count])
                count += 1
            dlist.append(dat_grid)
        d_grid = dlist[0]
        for l in range(1, 6):
            d_grid = np.concatenate((d_grid, dlist[l]), axis=0)
        d_grid = d_grid[:, :, np.newaxis]

        x.append(d_grid)
        if 'T' in ylist:
            y.append('T')
        else:
            y.append('R')

    x = np.array(x)
    y = np.array(y)

    f = open(dest_files[3], 'wb') #'Dataset/Hu/data_rpm_29_HU'
    pickle.dump((x, y), f)
    f.close()
