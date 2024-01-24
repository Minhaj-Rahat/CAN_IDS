from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten

# from keras.layers import activation
from keras import initializers

from keras.layers.core import Activation, Dense, Dropout
import yaml

bias_initializer = initializers.Constant(0.1)  # bias initialization for dense layer


def generate_cnn(num_classes, input_shape, config_file='can_config.yaml'):
    f = open(config_file, 'r')
    config = yaml.safe_load(f)

    filter_num_cnn = config['Hu']['train']['filter_num_cnn']
    kernel_size_cnn = config['Hu']['train']['kernel_size_cnn']
    # stride_cnn = config['CNN_HU']['stride_cnn']
    filter_num_pooling = config['Hu']['train']['filter_num_pooling']
    stride_pooling = config['Hu']['train']['stride_pooling']
    dropout = config['Hu']['train']['drop_out']

    model = Sequential()

    model.add(Conv2D(filter_num_cnn, kernel_size_cnn, padding='same', input_shape=input_shape))
    model.add(
        MaxPooling2D(pool_size=(filter_num_pooling, filter_num_pooling), strides=(stride_pooling, stride_pooling)))

    model.add(Flatten())
    model.add(Dense(128, bias_initializer=bias_initializer
                    ))
    model.add(Activation('tanh'))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model
