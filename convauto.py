from keras.layers import *
from keras.models import load_model, Sequential

encoding_dim = 32
image_x, image_y = (28, 28)
image_dim = image_x * image_y

input_img = Input(shape=(28, 28, 1))

from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), 28, 28, 1))
x_test = x_test.reshape((len(x_test), 28, 28, 1))

import csv
import os
from multiprocessing import Process, Lock, Manager

def run(psID, data, fnames, printlock, datalock):
    with printlock:
      print("Subprocess "+str(psID)+" starting!")
    try:
      while True:
        fname = fnames.pop()
        if fname.endswith(".csv"):
          with printlock:
            print(fname)
          with open("../csv/" + fname, "r") as file:
            try:
              file_frames = []
              while True:
                curr = len(file_frames)
                file_frames.append(bytearray())
                #with printlock:
                #  print "Starting", curr
                #  print len(data)
                while True:
                  line = next(file)
                  if len(line) <= 1:
                    break
                  file_frames[curr].extend(int(i) for i in line.split(','))
            except StopIteration:
              with datalock:
                data.extend(file_frames)
    except IndexError:
      with printlock:
        print("Subprocess "+str(psID)+" exiting!")

import pickle
try:
  print "Attempting to load rawdata.pickle..."
  with open("rawdata.pickle", "r") as f:
    data = pickle.load(f)
  print "Success"
except IOError:
  print "Failed, loading from ../csv..."
  with Manager() as manager:
    data = manager.list()
    fnames = manager.list(os.listdir("../csv"))
    printlock = Lock()
    datalock = Lock()

    nCores = 40
    processes = []
    for i in range(nCores):
      p = Process(target = run, args = (i,data,fnames,printlock,datalock))
      processes.append(p)

    for p in processes:
      p.start()
    for p in processes:
      p.join()

    data = list(data)

    print("Done! Pickling...")

    with open("rawdata.pickle", "wb") as p:
      pickle.dump(data, p, 2)

    print("Wrote rawdata.pickle")


from keras import backend as K
def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
    activations = get_activations([X_batch,0])
    return activations

try:
  print("Attempting to load model...")
  autoencoder = load_model('model.h5')
  print("Success.")
except IOError:
  print("Failed, training a model.")
  autoencoder = Sequential([
    Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28,28,1)),
    MaxPooling2D((2, 2), padding='same'),
    Conv2D(8, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), padding='same'),
    Conv2D(8, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), padding='same'),

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

    Conv2D(8, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(8, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(16, (3, 3), activation='relu'),
    UpSampling2D((2, 2)),
    Conv2D(1, (3, 3), activation='sigmoid', padding='same'),
  ])

  autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

  autoencoder.fit(x_train, x_train,
                epochs=5,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
  autoencoder.save('model.h5')

# encode and decode some digits
# note that we take them from the *test* set
#decoded_imgs = get_activations(autoencoder, -1, x_test)
decoded_imgs = autoencoder.predict(x_test)
# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(image_x, image_y))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(image_x, image_y))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
