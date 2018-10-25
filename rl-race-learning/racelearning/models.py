from typing import Tuple

from keras.layers import Activation, MaxPooling2D, Dropout, Convolution2D, Flatten, Dense
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras import regularizers

def create_vgg_like_model(input_shape: Tuple[int, int, int], output_units: int) -> Model:
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), border_mode='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, (3, 3), border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(output_units, activation='linear'))
    model.compile(optimizer='Adam', loss='mse', metrics=['mean_squared_error'])

    return model

#TUple changed
def create_atari_model(input_shape, output_units):
#def create_atari_model(input_shape: Tuple[ int, int, int], output_units: int) -> Model:
    model = Sequential()
    # model.add(Convolution2D(16, 5, 5, activation='relu', border_mode='same',
    #                         input_shape=input_shape, subsample=(3, 3), init ='glorot_uniform', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))
    # model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', subsample=(1, 1), init='glorot_uniform', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))
    # model.add(Flatten())
    # model.add(BatchNormalization())
    # model.add(Dense(256, activation='relu', init='glorot_uniform', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
    # model.add(Dense(output_units, activation='linear'))
    # model.compile(optimizer=optimizers.Nadam(), loss='mean_squared_error', metrics=['mean_squared_error'])

    model.add(Convolution2D(16, 8, 8, activation='relu', border_mode='same',
                              input_shape=input_shape, subsample=(4, 4)))
    model.add(Convolution2D(32, 4, 4, activation='relu', border_mode='same', subsample=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, input_shape=(2,), activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(output_units, activation='linear'))
    model.compile(optimizer='RMSprop', loss='mse', metrics=['mean_squared_error'])

    return model
