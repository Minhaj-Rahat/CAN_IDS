import result_estimate as re_es

# dos
model_path = 'models/Dos_Hu'
pos_in = 1  # attack index as positive class, in data it's index is 1
test_data = 'Dataset/Hu/Dos_test_file'
re_es.evaluate_model(model_path, test_data, pos_in, 'DoS')


# fuzzy
model_path = 'models/fuzzy_Hu'
pos_in = 1  # attack index as positive class, in data it's index is 1
test_data = 'Dataset/Hu/fuzzy_test_file'
re_es.evaluate_model(model_path, test_data, pos_in, 'Fuzzy')

# gear
model_path = 'models/gear_Hu'
pos_in = 1  # attack index as positive class, in data it's index is 1
test_data = 'Dataset/Hu/gear_test_file'
re_es.evaluate_model(model_path, test_data, pos_in, 'Gear')

# rpm
model_path = 'models/rpm_Hu'
pos_in = 1  # attack index as positive class, in data it's index is 1
test_data = 'Dataset/Hu/rpm_test_file'
re_es.evaluate_model(model_path, test_data, pos_in, 'RPM')