from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.datasets import mnist
import numpy as np

import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt


inputs = Input(shape=(300, 336, 1))

x = Convolution2D(64, 15, 15, activation='relu', border_mode='same')(input_img) #nb_filter, nb_row, nb_col
x = MaxPooling2D((10, 10), border_mode='same')(x)
x = Convolution2D(32, 15, 15, activation='relu', border_mode='same')(x)
x = MaxPooling2D((10, 10), border_mode='same')(x)
x = Convolution2D(32, 15, 15, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((10, 10), border_mode='same')(x)

autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 300, 336, 1))
x_test = np.reshape(x_test, (len(x_test), 300, 336, 1))


autoencoder.fit(x_train, x_train,
        epochs=10,
        batch_size=128,
        shuffle=True,
        validation_data=(x_test, x_test),
        callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
decoded_imgs = autoencoder.predict(x_test)

n=10
plt.figure(figsize=(20,4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(330, 336))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #display reconstruction
    ax = plt.subplot(2, n, i+n +1)
    plt.imshow(decoded_imgs[i].reshape(330,336))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
