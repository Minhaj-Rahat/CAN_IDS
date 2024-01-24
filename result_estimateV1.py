import keras
import pickle
import numpy as np


def evaluate_model(model_path, test_data_file, positive_index):
    model = keras.models.load_model(model_path)
    f = open(test_data_file, 'rb')
    test_data = pickle.load(f)
    f.close()
    x_test, y_test = test_data[0], test_data[1]

    Tp = 0
    Fp = 0
    Tn = 0
    Fn = 0

    total_sample = len(y_test)
    for i in range(total_sample):
        pred = model.predict(x_test[i][np.newaxis, :, :, :], verbose=0)
        if np.argmax(y_test[i] == positive_index):
            if np.argmax(pred) == np.argmax(y_test[i]):
                Tp += 1
            else:
                Fn += 1
        else:
            if np.argmax(pred) == np.argmax(y_test[i]):
                Tn += 1
            else:
                Fp += 1

    print(f'Tp: {Tp}, Tn: {Tn}, Fp: {Fp}, Fn: {Fn}')

    recall = Tp / (Tp + Fn)
    Fnr = Fn / (Tp + Fn)
    precision = Tp / (Tp + Fp)
    F1 = (2 * precision * recall) / (precision + recall)
    print(f'Recall: {recall} False Negative Rate: {Fnr}, Precision: {precision}, F1_score: {F1}')


# test
model_path = 'models/Dos_Hu'
pos_in = 1  # attack index as positive class, in data it's index is 1
test_data = 'Dataset/Hu/Dos_test_file'
evaluate_model(model_path, test_data, pos_in)
