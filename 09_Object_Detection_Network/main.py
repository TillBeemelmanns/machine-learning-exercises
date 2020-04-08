import os
import numpy as np

from glob import glob

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy, sparse_categorical_crossentropy

from skimage.transform import resize
from skimage import io

BATCHES_PER_EPOCH = 50


def pokemon_generator_multiclass(batch_size=64):
    # generate image and targets
    while True:
        # Each epoch will have 50 batches. Why? No reason
        for _ in range(BATCHES_PER_EPOCH):
            X = np.zeros((batch_size, IMAGE_DIM, IMAGE_DIM, 3))
            y = np.zeros((batch_size, 8))

            for i in range(batch_size):
                # select a random background
                bg_idx = np.random.choice(len(backgrounds))
                bg = backgrounds[bg_idx]
                bg_h, bg_w, _ = bg.shape
                rnd_h = np.random.randint(bg_h - IMAGE_DIM)
                rnd_w = np.random.randint(bg_w - IMAGE_DIM)
                X[i] = bg[rnd_h:rnd_h + IMAGE_DIM, rnd_w:rnd_w + IMAGE_DIM].copy()

                # 25% no object, 25% + 25% + 25% for the 3 classes
                appear = (np.random.random() < 0.75)
                if appear:

                    # choose a pokemon
                    pk_idx = np.random.randint(3)
                    pk, h, w, _ = poke_data[pk_idx]

                    # resize object - make it bigger or smaller
                    scale = 0.5 + np.random.random()
                    new_height = int(h * scale)
                    new_width = int(w * scale)
                    obj = resize(
                        pk,
                        (new_height, new_width),
                        preserve_range=True).astype(np.uint8)  # keep it from 0..255

                    # maybe flip
                    if np.random.random() < 0.5:
                        obj = np.fliplr(obj)

                    # choose a random location to store the object
                    row0 = np.random.randint(IMAGE_DIM - new_height)
                    col0 = np.random.randint(IMAGE_DIM - new_width)
                    row1 = row0 + new_height
                    col1 = col0 + new_width

                    # can't 'just' assign obj to a slice of X
                    # since the transparent parts will be black (0)
                    mask = (obj[:, :, 3] == 0)  # find where the pokemon is 0
                    bg_slice = X[i, row0:row1, col0:col1, :]  # where we want to place `obj`
                    bg_slice = np.expand_dims(mask, -1) * bg_slice  # (h,w,1) x (h,w,3)
                    bg_slice += obj[:, :, :3]  # "add" the pokemon to the slice
                    X[i, row0:row1, col0:col1, :] = bg_slice  # put the slice back

                    # make targets

                    # location
                    y[i, 0] = row0 / IMAGE_DIM
                    y[i, 1] = col0 / IMAGE_DIM
                    y[i, 2] = (row1 - row0) / IMAGE_DIM
                    y[i, 3] = (col1 - col0) / IMAGE_DIM

                    # class
                    y[i, 4 + pk_idx] = 1

                # did the pokemon appear?
                y[i, 7] = appear

            yield X / 255., y


# Make predictions
def pokemon_prediction_multiclass():
    # select a random background
    bg_idx = np.random.choice(len(backgrounds))
    bg = backgrounds[bg_idx]
    bg_h, bg_w, _ = bg.shape
    rnd_h = np.random.randint(bg_h - IMAGE_DIM)
    rnd_w = np.random.randint(bg_w - IMAGE_DIM)
    x = bg[rnd_h:rnd_h + IMAGE_DIM, rnd_w:rnd_w + IMAGE_DIM].copy()

    appear = (np.random.random() < 0.75)
    if appear:

        # choose a pokemon
        pk_idx = np.random.randint(3)
        pk, h, w, _ = poke_data[pk_idx]

        # resize charmander - make it bigger or smaller
        scale = 0.5 + np.random.random()
        new_height = int(h * scale)
        new_width = int(w * scale)
        obj = resize(
            pk,
            (new_height, new_width),
            preserve_range=True).astype(np.uint8)  # keep it from 0..255

        # maybe flip
        if np.random.random() < 0.5:
            obj = np.fliplr(obj)

        # choose a random location to store the object
        row0 = np.random.randint(IMAGE_DIM - new_height)
        col0 = np.random.randint(IMAGE_DIM - new_width)
        row1 = row0 + new_height
        col1 = col0 + new_width

        # can't 'just' assign obj to a slice of X
        # since the transparent parts will be black (0)
        mask = (obj[:, :, 3] == 0)  # find where the pokemon is 0
        bg_slice = x[row0:row1, col0:col1, :]  # where we want to place `obj`
        bg_slice = np.expand_dims(mask, -1) * bg_slice  # (h,w,1) x (h,w,3)
        bg_slice += obj[:, :, :3]  # "add" the pokemon to the slice
        x[row0:row1, col0:col1, :] = bg_slice  # put the slice back
        actual_class = class_names[pk_idx]
        print("true:", row0, col0, row1, col1, actual_class)

    # Predict
    X = np.expand_dims(x, 0) / 255.
    p = model.predict(X)[0]

    # Plot
    fig, ax = plt.subplots(1)
    ax.imshow(x.astype(np.uint8))

    # Draw the box
    if p[-1] > 0.5:
        row0 = int(p[0] * IMAGE_DIM)
        col0 = int(p[1] * IMAGE_DIM)
        row1 = int(row0 + p[2] * IMAGE_DIM)
        col1 = int(col0 + p[3] * IMAGE_DIM)
        class_pred_idx = np.argmax(p[4:7])
        class_pred = class_names[class_pred_idx]
        print("pred:", row0, col0, row1, col1, class_pred)
        rect = Rectangle(
            (p[1] * IMAGE_DIM, p[0] * IMAGE_DIM),
            p[3] * IMAGE_DIM, p[2] * IMAGE_DIM, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    else:
        print("pred: no object")
    plt.show()



def download_data():
    if not os.path.isdir("images"):
        os.mkdir("images")
        os.system("wget -nc https://lazyprogrammer.me/course_files/charmander-tight.png -P images")
        os.system("wget -nc https://lazyprogrammer.me/course_files/bulbasaur-tight.png -P images")
        os.system("wget -nc https://lazyprogrammer.me/course_files/squirtle-tight.png -P images")
        os.system("wget -nc https://lazyprogrammer.me/course_files/backgrounds.zip -P images")
        os.system("unzip -n images/backgrounds.zip -d images")


def custom_loss(y_true, y_pred):
    # target is a 8-tuple
    # (row, col, height, width, class1, class2, class3, object_appeared)

    bce = binary_crossentropy(y_true[:, :4], y_pred[:, :4])  # bounding box
    cce = categorical_crossentropy(y_true[:, 4:7], y_pred[:, 4:7])  # object class
    bce2 = binary_crossentropy(y_true[:, -1], y_pred[:, -1])  # object appeared

    return bce * y_true[:, -1] + cce * y_true[:, -1] + 0.5 * bce2


def make_model():
    vgg = tf.keras.applications.VGG16(
        input_shape=[IMAGE_DIM, IMAGE_DIM, 3],
        include_top=False,
        weights='imagenet')
    x = Flatten()(vgg.output)
    bounding_box = Dense(4, activation='sigmoid')(x)  # bounding box (x, y, height, width)
    classification = Dense(3, activation='softmax')(x)  # object class (class1, class2, class3)
    object_appeared = Dense(1, activation='sigmoid')(x)  # object appeared (object_appeared)
    x = Concatenate()([bounding_box, classification, object_appeared])

    model = Model(vgg.input, x)
    model.compile(loss=custom_loss, optimizer=Adam(lr=0.0001))
    return model


if __name__ == '__main__':
    IMAGE_DIM = 200

    download_data()

    backgrounds = []
    background_files = glob('images/backgrounds/*.jpg')
    for f in background_files:
        # Note: they may not all be the same size
        bg = np.array(image.load_img(f))
        backgrounds.append(bg)

    ch = io.imread('images/charmander-tight.png')
    bb = io.imread('images/bulbasaur-tight.png')
    sq = io.imread('images/squirtle-tight.png')

    ch = np.array(ch)
    bb = np.array(bb)
    sq = np.array(sq)

    CH_H, CH_W, CH_C = ch.shape
    BB_H, BB_W, BB_C = bb.shape
    SQ_H, SQ_W, SQ_C = sq.shape

    poke_data = [
        [ch, CH_H, CH_W, CH_C],
        [bb, BB_H, BB_W, BB_C],
        [sq, SQ_H, SQ_W, SQ_C],
    ]
    class_names = ['Charmander', 'Bulbasaur', 'Squirtle']

    xx, yy = None, None
    for x, y in pokemon_generator_multiclass():
        xx, yy = x, y
        break

    n = yy.shape[0]
    idx = np.random.randint(n)

    fig, ax = plt.subplots(1)
    plt.imshow(xx[idx])
    rect = Rectangle(
        (yy[idx][1] * IMAGE_DIM, yy[idx][0] * IMAGE_DIM),
         yy[idx][3] * IMAGE_DIM, yy[idx][2] * IMAGE_DIM,
        linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()

    exit()

    model = make_model()
    model.fit_generator(
        pokemon_generator_multiclass(),
        steps_per_epoch=BATCHES_PER_EPOCH,
        epochs=3,
    )

    pokemon_prediction_multiclass()

