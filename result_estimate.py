import keras
import pickle
import numpy as np


def evaluate_model(model_path, test_data_file, positive_index, attack):
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

    y_pred = model.predict(x_test, batch_size=64, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)

    for i in range(total_sample):

        if np.argmax(y_test[i] == positive_index):
            if y_pred_bool[i] == np.argmax(y_test[i]):
                Tp += 1
            else:
                Fn += 1
        else:
            if y_pred_bool[i] == np.argmax(y_test[i]):
                Tn += 1
            else:
                Fp += 1

    print(f'Attack : {attack}')

    print(f'Tp: {Tp}, Tn: {Tn}, Fp: {Fp}, Fn: {Fn}')

    accuracy = (Tp+Tn)/total_sample
    recall = Tp / (Tp + Fn)
    Fnr = Fn / (Tp + Fn)
    precision = Tp / (Tp + Fp)
    F1 = (2 * precision * recall) / (precision + recall)
    print(f'Accuracy: {accuracy} Recall: {recall} False Negative Rate: {Fnr}, Precision: {precision}, F1_score: {F1}')


# test

