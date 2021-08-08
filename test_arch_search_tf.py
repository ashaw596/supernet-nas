from unittest import TestCase

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from arch_search_tf import MixedModuleTf, SupernetTemperatureCallback, NasModel, SupernetArchWatcherCallback


class TestMixedModuleTf(TestCase):
    def test_build(self):
        # resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
        # tf.config.experimental_connect_to_cluster(resolver)
        # tf.tpu.experimental.initialize_tpu_system(resolver)
        # strategy = tf.distribute.TPUStrategy(resolver)

        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")

        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        num_classes = 10
        input_shape = (28, 28, 1)

        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        print("x_train shape:", x_train.shape)
        print(x_train.shape[0], "train samples")
        print(x_test.shape[0], "test samples")

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        with strategy.scope():
            model = keras.Sequential(
                [
                    keras.layers.InputLayer(input_shape=input_shape),
                    # layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                    MixedModuleTf({
                        '3x3': layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
                        '1x1': layers.Conv2D(32, kernel_size=(1, 1), activation="relu", padding="same")
                    }, cost_loss_multiplier=1e-6),
                    layers.MaxPooling2D(pool_size=(2, 2)),
                    MixedModuleTf([
                        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                        layers.Conv2D(64, kernel_size=(3, 3), activation="relu")
                    ]),
                    layers.MaxPooling2D(pool_size=(2, 2)),
                    layers.Flatten(),
                    layers.Dropout(0.5),
                    # MyDenseLayer(num_classes)
                    MixedModuleTf([
                        layers.Dense(num_classes, activation="softmax"),
                        layers.Dense(num_classes, activation="softmax"),
                    ])
                ]
            )

            arch_optim = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0)

            model = NasModel(model)
            model.build([None] + list(input_shape))

            model.summary()
            batch_size = 128

            callbacks = [
                SupernetTemperatureCallback(model, start_epoch=2, final_epoch=5, start_temp=5, end_temp=1),
                SupernetArchWatcherCallback(model),
            ]

            model.compile(loss="categorical_crossentropy",
                          optimizer="adam",
                          metrics=["accuracy"],
                          arch_optimizer=arch_optim)
        model.fit(x_train, y_train,
                  steps_per_epoch=10,
                  batch_size=batch_size,
                  epochs=10,
                  validation_split=0.1,
                  callbacks=callbacks)
        # self.fail()
