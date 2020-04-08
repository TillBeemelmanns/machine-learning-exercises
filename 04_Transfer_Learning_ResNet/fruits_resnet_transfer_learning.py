from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils.utils import plot_confusion_matrix

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from glob import glob


def main():
    # re-size all the images to this
    IMAGE_SIZE = [100, 100]

    # training config:
    epochs = 5
    batch_size = 32

    train_path = '../data/fruits-360-small/Training'
    val_path = '../data/fruits-360-small/Test'

    # useful for getting number of files
    image_files = glob(train_path + '/*/*.jp*g')
    valid_image_files = glob(val_path + '/*/*.jp*g')

    # useful for getting number of classes
    folders = glob(train_path + '/*')

    num_classes = len(folders)

    # add preprocessing layer to the front of VGG
    resnet50v2 = ResNet50V2(input_shape=IMAGE_SIZE + [3],
                            weights='imagenet',
                            include_top=False)

    # don't train existing weights
    for layer in resnet50v2.layers:
        layer.trainable = False

    # our layers - you can add more if you want
    x = Flatten()(resnet50v2.output)
    prediction = Dense(num_classes, activation='softmax')(x)

    # create a model object
    model = Model(inputs=resnet50v2.input, outputs=prediction)

    # view the structure of the model
    model.summary()

    # tell the model what cost and optimization method to use
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # create an instance of ImageDataGenerator
    gen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=preprocess_input
    )

    # get label mapping for confusion matrix plot later
    test_gen = gen.flow_from_directory(val_path, target_size=IMAGE_SIZE)

    print(test_gen.class_indices)

    labels = [None] * len(test_gen.class_indices)
    for k, v in test_gen.class_indices.items():
        labels[v] = k

    # should be a strangely colored image (due to VGG weights being BGR)
    for x, y in test_gen:
        print(x[0].size)
        print("min:", x[0].min(), "max:", x[0].max())
        plt.title(labels[np.argmax(y[0])])
        plt.imshow(x[0])
        plt.show()
        break

    # create generators
    train_generator = gen.flow_from_directory(
        train_path,
        target_size=IMAGE_SIZE,
        shuffle=True,
        batch_size=batch_size,
    )

    valid_generator = gen.flow_from_directory(
        val_path,
        target_size=IMAGE_SIZE,
        shuffle=True,
        batch_size=batch_size,
    )

    r = model.fit_generator(
        generator=train_generator,
        validation_data=valid_generator,
        epochs=epochs,
        steps_per_epoch=len(image_files) // batch_size,
        validation_steps=len(valid_image_files) // batch_size,
    )

    def get_confusion_matrix(data_path, N):
        # we need to see the data in the same order
        # for both predictions and targets
        print("Generating confusion matrix", N)
        predictions = []
        targets = []
        i = 0
        for x, y in gen.flow_from_directory(data_path, target_size=IMAGE_SIZE, shuffle=False,
                                            batch_size=batch_size * 2):
            i += 1
            if i % 50 == 0:
                print(i)
            p = model.predict(x)
            p = np.argmax(p, axis=1)
            y = np.argmax(y, axis=1)
            predictions = np.concatenate((predictions, p))
            targets = np.concatenate((targets, y))
            if len(targets) >= N:
                break

        cm = confusion_matrix(targets, predictions)
        return cm

    train_cm = get_confusion_matrix(train_path, len(image_files))
    print(train_cm)
    val_cm = get_confusion_matrix(val_path, len(valid_image_files))
    print(val_cm)

    # plot some data
    # loss
    plt.plot(r.history['loss'], label='train loss')
    plt.plot(r.history['val_loss'], label='val loss')
    plt.legend()
    plt.show()

    # accuracies
    plt.plot(r.history['accuracy'], label='train acc')
    plt.plot(r.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.show()

    plot_confusion_matrix(train_cm, labels, title='Train confusion matrix')
    plot_confusion_matrix(val_cm, labels, title='Validation confusion matrix')


if __name__ == '__main__':
    main()
