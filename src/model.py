import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

from keras.models import load_model
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau, CSVLogger
from keras.optimizers import Adam

from keras import backend as K
from distutils.version import LooseVersion as LV
from keras import __version__
print('Using Keras version:', __version__, 'backend:', K.backend())
assert(LV(__version__) >= LV("2.0.0"))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import cv2
import pickle

from src.define_model import define_model
from src.constants import *

class Model:

    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        return


    def load_data(self,data_path=None):

        """Loading the EMINST dataset"""
        if data_path is None:
            data = loadmat(os.path.abspath(os.path.join(os.getcwd(), DATA_PATH)))
        else:
            data = loadmat(os.path.abspath(os.path.join(os.getcwd(), data_path)))

        # Loading Training Data
        X_train = data["dataset"][0][0][0][0][0][0]
        y_train = data["dataset"][0][0][0][0][0][1]
        X_train = X_train.astype('float32')
        X_train /= 255.0

        ##Loading Testing Data
        X_test = data["dataset"][0][0][1][0][0][0]
        y_test = data["dataset"][0][0][1][0][0][1]
        X_test = X_test.astype('float32')
        X_test /= 255.0

        # one-hot encoding:
        Y_train = np_utils.to_categorical(y_train, NUM_CLASSES)
        Y_test = np_utils.to_categorical(y_test, NUM_CLASSES)

        # input image dimensions
        img_rows, img_cols = 28, 28


        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

        print('Intermediate X_train:', X_train.shape)
        print('Intermediate X_test:', X_test.shape)

        # Reshaping all images into 28*28 for pre-processing
        X_train = X_train.reshape(X_train.shape[0], 28, 28)
        X_test = X_test.reshape(X_test.shape[0], 28, 28)

        # for train data
        for t in range(X_train.shape[0]):
            X_train[t] = np.transpose(X_train[t])

        # for test data
        for t in range(X_test.shape[0]):
            X_test[t] = np.transpose(X_test[t])

        print('Process Complete: Rotated and reversed test and train images!')

        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_train = X_train.reshape(X_train.shape[0], 784, )

        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 784, )

        print('EMNIST data loaded: train:', len(X_train), 'test:', len(X_test))
        print('Flattened X_train:', X_train.shape)
        print('Y_train:', Y_train.shape)
        print('Flattened X_test:', X_test.shape)
        print('Y_test:', Y_test.shape)
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    def character_model(self):

        self.model = define_model(NUM_CLASSES)

    def loadmodel(self,path=None):
        if path is not None and os.path.exists(path):
            self.model = load_model(path)
        else:
            if path is None and os.path.exists(LOAD_MODEL_NAME):
                self.model = load_model(LOAD_MODEL_NAME)
        return


    def train(self):

        cb_checkpoint = ModelCheckpoint(CHECKPOINT_PATH, verbose=1, save_weights_only=False, period=1)
        cb_early_stopper = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE)
        reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=20, verbose=1)
        csv_logger = CSVLogger(LOG_FILE)

        callback_values = [cb_checkpoint,cb_early_stopper,reduce_on_plateau,csv_logger]

        if LOAD_MODEL:
            self.loadmodel()

        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        history = self.model.fit(self.X_train, self.Y_train, validation_split=0.1,
                                 epochs=EPOCH, callbacks=callback_values,batch_size=BATCH_SIZE)

        plt.figure(figsize=(5,3))
        plt.plot(history.epoch,history.history['loss'])
        plt.title('loss')

        plt.figure(figsize=(5,3))
        plt.plot(history.epoch,history.history['acc'])
        plt.title('accuracy')


    def test(self, img_path=None):
        try:
            with open(os.path.abspath(os.path.join(os.getcwd(), "./data/mapping.pkl"))) as f:
                mapping = pickle.load(f)
        except Exception as e:
            print(e)
            mapping = ['0', '1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

        if self.model is None:
            print("No model found")
            exit()

        if img_path is None:
            print("Image path wrong. Loading from default path")
            img_path = TEST_PATH

        image = cv2.imread(img_path)
        image = cv2.resize(image, (28, 28),interpolation=cv2.INTER_CUBIC)
        print("Image shape {}".format(image.shape))

        # grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # binary
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((3, 3), np.uint8)

        # dilation
        pred_img = cv2.dilate(thresh, kernel, iterations=1)

        #erode
        pred_img = cv2.erode(pred_img, kernel, iterations=1)

        pred_img = pred_img/255.0
        # plt.imshow(pred_img,cmap="gray")
        # plt.show()

        pred_test_img = pred_img
        pred_test_img = pred_test_img.reshape(1,784)
        prediction = mapping[np.argmax(self.model.predict(pred_test_img), axis=1)[0]]
        print("Predicted Value : {}".format(prediction))
