
"""
# This script is an easy implementation of a LeNet-5 architecture
from http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf. It has minor
modifications to the proposed architecture.

"""

from snare import Snare
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras.datasets import mnist


def prepare_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    # Scale all values
    x_train /= 255
    x_test /= 255

    # One-hot encoding of labels
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Reshape input images
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    return (x_train, y_train), (x_test, y_test)


# Define model architecture
lenet5 = Sequential()

lenet5.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1),
                         activation="tanh", input_shape=(28, 28, 1),
                         padding="same"))
lenet5.add(layers.AveragePooling2D(pool_size=(
    2, 2), strides=(2, 2), padding="valid"))
lenet5.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(
    1, 1), activation="tanh", padding="valid"))
lenet5.add(layers.AveragePooling2D(pool_size=(
    2, 2), strides=(2, 2), padding="valid"))

lenet5.add(layers.Flatten())
lenet5.add(layers.Dense(units=120, activation="tanh"))
lenet5.add(layers.Dense(84, activation="tanh"))
lenet5.add(layers.Dense(10, activation="softmax"))

# Compile model
compile_args = {'loss': losses.categorical_crossentropy,
                'optimizer': 'SGD', 'metrics': ['accuracy']}
lenet5.compile(**compile_args)
lenet5.summary()

# Train model
batch_size = 128
dataset = prepare_dataset()
(x_train, y_train), (x_test, y_test) = dataset
lenet5.fit(x=x_train, y=y_train, epochs=200, batch_size=batch_size,
           validation_data=(x_test, y_test), verbose=1)

snare = Snare(lenet5, compile_args)
result = snare.reduce(dataset, 0.005)

test_score = result.evaluate(x_test, y_test)
print("LeNet5: loss: {:.4f}, accuracy: {:.4f}".format(
    test_score[0], test_score[1]))
