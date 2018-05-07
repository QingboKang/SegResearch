import random, cv2, os
import numpy as np
from keras.utils import np_utils


def generate(data_type, X, nb_classes, batch_size, img_channels, row, col, source_path):
    index = 0
    while True:
        if index == 0 and data_type == 'train':
            random.shuffle(X)

        current_x = X[index : index+batch_size]

        # update index for next batch output
        if index + batch_size >= len(X):
            index = 0
        else:
            index += batch_size

        batch_x = np.empty((len(current_x), row, col, img_channels), dtype='float32')
        batch_y = np.empty((len(current_x),), dtype="uint8")

        for i in range(len(current_x)):
            # orig_1_76_0.png
            label = int(current_x[i].split('.')[0][-1])
            image_orig = cv2.imread(os.path.join(source_path, current_x[i]), cv2.IMREAD_COLOR)
            image_ws = cv2.imread(os.path.join(source_path, current_x[i].replace('orig', 'watershed')), cv2.IMREAD_GRAYSCALE)

            batch_x[i, :, :, :] = np.concatenate([image_orig, np.expand_dims(image_ws, axis=-1)], axis=-1)
            batch_y[i] = label

        batch_y = np_utils.to_categorical(batch_y, nb_classes)

        yield batch_x, batch_y
