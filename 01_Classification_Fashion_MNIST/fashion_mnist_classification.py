import numpy as np

from mnist import MNIST

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D,\
                                    Flatten, BatchNormalization, Activation


def get_model(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def main():
    mndata = MNIST('../data/mnist_fasion', return_type="numpy")

    X_train, y_train = mndata.load_training()
    X_test, y_test = mndata.load_testing()

    img_rows, img_cols = 28, 28
    num_classes = 10

    X_train = X_train.reshape(-1, img_rows, img_cols, 1)
    X_test = X_test.reshape(-1, img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    mean = np.mean(X_train)
    std = np.std(X_train)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    print('x_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    model = get_model(input_shape, num_classes)

    model.fit(X_train,
              y_train,
              epochs=2,
              batch_size=32,
              verbose=1)

    score = model.evaluate(X_test,
                           y_test,
                           batch_size=128)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    main()
