import numpy as np
import yaml
import CNN_HU as cnn_hu
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.utils import np_utils

import pickle
def train(model_files, test_files, config_data,author,dest_files):
    # Dos training
    batch_size = 64  # experimental
    epoch = 2  # experimental
    train_data = 0.75
    test_data = 0.25
    num_classes = 2

    loss = BinaryCrossentropy(from_logits=True)
    lr = config_data[author]['train']['lr']
    opt = Adam(learning_rate=lr)

    model_path = model_files[0] #dos
    test_file = test_files[0]  # Dos
    mosaic_size = 16  # for dos it is 16, fuzzy 48, gear 36, rpm 36




    input_shape = (mosaic_size, mosaic_size, 1)
    # load data
    print('Loading Data Dos---')
    data_file = dest_files[0] #dos
    f = open(data_file, 'rb')
    data = pickle.load(f)
    f.close()
    X = data[0]
    Y = data[1]
    print(f'X shape: {X.shape}')
    print(f'Y shape: {Y.shape}')

    # create binary class label for Y
    y = []
    for i in Y:
        if i == 'R':
            y.append(0)
        else:
            y.append(1)


    y = np.array(y)

    print(f'Y shape: {y.shape}')

    # generate train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_data, shuffle=True, random_state=25)
    print(f'Split size: Train: {X_train.shape} - {y_train.shape}, Test: {X_test.shape} - {y_test.shape}')

    # converting classes to categorical classes
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    # dump test file
    test_data = (X_test, y_test)
    f = open(test_file, 'wb')
    pickle.dump(test_data, f)
    f.close()

    # create the cnn model
    print('Building the CNN model dor Dos:')
    model = cnn_hu.generate_cnn(num_classes, input_shape)
    model.compile(loss=loss, optimizer=opt, metrics='accuracy')
    print('Model has been built')

    # Trainig session
    info = model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch)
    print('Dos Training Done. Saving the model.....')

    # save the model
    model.save(model_path)
    print('Dos Model saved')

    #clear memory
    del X_train, X_test, y_train, y_test, model


    # train fuzzy model
    model_path = model_files[1]  # fuzzy
    test_file = test_files[1]  # fuzzy
    mosaic_size = 48  # for dos it is 16, fuzzy 48, gear 36, rpm 36

    input_shape = (mosaic_size, mosaic_size, 1)
    # load data
    print('Loading Data Fuzzy---')
    data_file = dest_files[1]  # fuzzy
    f = open(data_file, 'rb')
    data = pickle.load(f)
    f.close()
    X = data[0]
    Y = data[1]
    print(f'X shape: {X.shape}')
    print(f'Y shape: {Y.shape}')

    # create binary class label for Y
    y = []
    for i in Y:
        if i == 'R':
            y.append(0)
        else:
            y.append(1)


    y = np.array(y)

    print(f'Y shape: {y.shape}')

    # generate train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_data, shuffle=True, random_state=25)
    print(f'Split size: Train: {X_train.shape} - {y_train.shape}, Test: {X_test.shape} - {y_test.shape}')

    # converting classes to categorical classes
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    # dump test file
    test_data = (X_test, y_test)
    f = open(test_file, 'wb')
    pickle.dump(test_data, f)
    f.close()

    # create the cnn model
    print('Building the CNN model for Fuzzy:')
    model = cnn_hu.generate_cnn(num_classes, input_shape)
    model.compile(loss=loss, optimizer=opt, metrics='accuracy')
    print('Model has been built')

    # Trainig session
    info = model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch)
    print('Training Done. Saving the model.....')

    # save the model
    model.save(model_path)
    print('Fuzzy  Model saved')

    # clear memory
    del X_train, X_test, y_train, y_test, model

    model_path = model_files[2]  # gear
    test_file = test_files[2]  # gear
    mosaic_size = 36  # for dos it is 16, fuzzy 48, gear 36, rpm 36

    input_shape = (mosaic_size, mosaic_size, 1)
    # load data
    print('Loading Data Gear---')
    data_file = dest_files[2]  # gear
    f = open(data_file, 'rb')
    data = pickle.load(f)
    f.close()
    X = data[0]
    Y = data[1]
    print(f'X shape: {X.shape}')
    print(f'Y shape: {Y.shape}')

    # create binary class label for Y
    y = []
    for i in Y:
        if i == 'R':
            y.append(0)
        else:
            y.append(1)


    y = np.array(y)

    print(f'Y shape: {y.shape}')

    # generate train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_data, shuffle=True, random_state=25)
    print(f'Split size: Train: {X_train.shape} - {y_train.shape}, Test: {X_test.shape} - {y_test.shape}')

    # converting classes to categorical classes
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    # dump test file
    test_data = (X_test, y_test)
    f = open(test_file, 'wb')
    pickle.dump(test_data, f)
    f.close()

    # create the cnn model
    print('Building the CNN model for Gear:')
    model = cnn_hu.generate_cnn(num_classes, input_shape)
    model.compile(loss=loss, optimizer=opt, metrics='accuracy')
    print('Model has been built')

    # Trainig session
    info = model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch)
    print('Training Done. Saving the model.....')

    # save the model
    model.save(model_path)
    print('Gear Model saved')

    # clear memory
    del X_train, X_test, y_train, y_test, model

    model_path = model_files[3]  # rpm
    test_file = test_files[3]  # rpm
    mosaic_size = 36  # for dos it is 16, fuzzy 48, gear 36, rpm 36

    input_shape = (mosaic_size, mosaic_size, 1)
    # load data
    print('Loading Data RPM---')
    data_file = dest_files[3]  # rpm
    f = open(data_file, 'rb')
    data = pickle.load(f)
    f.close()
    X = data[0]
    Y = data[1]
    print(f'X shape: {X.shape}')
    print(f'Y shape: {Y.shape}')

    # create binary class label for Y
    y = []
    for i in Y:
        if i == 'R':
            y.append(0)
        else:
            y.append(1)


    y = np.array(y)

    print(f'Y shape: {y.shape}')

    # generate train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_data, shuffle=True, random_state=25)
    print(f'Split size: Train: {X_train.shape} - {y_train.shape}, Test: {X_test.shape} - {y_test.shape}')

    # converting classes to categorical classes
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    # dump test file
    test_data = (X_test, y_test)
    f = open(test_file, 'wb')
    pickle.dump(test_data, f)
    f.close()

    # create the cnn model
    print('Building the CNN model for RPM:')
    model = cnn_hu.generate_cnn(num_classes, input_shape)
    model.compile(loss=loss, optimizer=opt, metrics='accuracy')
    print('Model has been built')

    # Trainig session
    info = model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch)
    print('Training Done. Saving the model.....')

    # save the model
    model.save(model_path)
    print('RPM Model saved')

    # clear memory
    del X_train, X_test, y_train, y_test, model