import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import to_categorical
import keras.metrics
import tensorflow as tf

import time

if __name__ == '__main__':
    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Create dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).repeat()
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).repeat()

    # Create model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(784,)),
        Dense(10, activation='softmax'),
    ])

    # Compile model
    model.compile(
        loss='mean_squared_error',
        optimizer=SGD(lr=0.01)
    )
    batch_size = 32

    # Train model for 60k steps
    start_time = time.time()
    model.fit(
        train_dataset.batch(batch_size),
        epochs=1,
        steps_per_epoch=60000,
        verbose=1,
    )
    end_time = time.time()
    print('Elapsed time: {}s'.format(end_time - start_time))

    # Evaluate model (accuracy)
    test_it = iter(test_dataset.batch(batch_size))
    n_correct = 0
    n_test_steps = 10000
    for i in range(n_test_steps):
        X, Y = next(test_it)
        logits = model(X, training=False)
        y_hat = tf.argmax(logits, axis=1)
        y = tf.argmax(Y, axis=1)
        n_correct += tf.reduce_sum(tf.cast(tf.equal(y, y_hat), tf.int32))

    print('Accuracy: {}'.format(n_correct / n_test_steps / batch_size))
