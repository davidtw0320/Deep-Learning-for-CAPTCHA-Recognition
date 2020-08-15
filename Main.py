import numpy as np
from keras.models import Model
from keras.utils import np_utils
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import Adam
from Preprocessing import *
from time import time

def getModle(input):
    x = input
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_1_1')(x)
    x = Conv2D(32, (3, 3), activation='relu', name='conv_1_2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2_1')(x)
    x = Conv2D(64, (3, 3), activation='relu', name='conv_2_2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_3_1')(x)
    x = Conv2D(128, (3, 3), activation='relu', name='conv_3_2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)
    x = Conv2D(256, (3, 3), activation='relu',name='conv_4_1')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    x = [Dense(characterNumber, name='digit1', activation='softmax')(x),
        Dense(characterNumber, name='digit2', activation='softmax')(x),
        Dense(characterNumber, name='digit3', activation='softmax')(x),
        Dense(characterNumber, name='digit4', activation='softmax')(x),
        Dense(characterNumber, name='digit5', activation='softmax')(x),
        Dense(characterNumber, name='digit6', activation='softmax')(x)]
    return Model(inputs=input, outputs=x)


xTrainShape = getImageSize('train/data01_train')
(xTrain, yTrain)= load_data(xTrainPath='train/data01_train', yTrainPath='train/data01_train.csv')
xTrain = xTrain/255.
yTrain = np_utils.to_categorical(yTrain, num_classes=characterNumber).tolist()

tensorBoard = TensorBoard(log_dir = "./logs", histogram_freq = 1)
model = getModle(Input((xTrainShape[0], xTrainShape[1], xTrainShape[2])))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(xTrain, yTrain, batch_size=400, epochs=150, verbose=1 ,callbacks=[tensorBoard])  
model.save('model/data01_model_v8.h5')