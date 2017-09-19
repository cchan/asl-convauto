from keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train,
        epochs=50,
        batch_size=128,
        shuffle=True,
        validation_data=(x_test, x_test),
        callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
