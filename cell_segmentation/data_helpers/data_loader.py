import os, cv2, random
import fnmatch
import numpy as np


# this will return a random split of data for train, val and test
def load_source(source_path):
    total_size = len(fnmatch.filter(os.listdir(source_path), '*.bmp'))

    # todo: this should be randomly selevted when data is ready
    train_index = [1]
    val_index = [1]
    test_index = [1]

    train_source = []
    val_source = []
    test_source = []

    idx = 0
    for img in os.listdir(source_path):
        # orig_1_76_0.png
        if img.split('_')[0] != 'orig':
            continue

        if int(img.split('_')[1]) in train_index:
            train_source.append(img)
        if int(img.split('_')[1]) in val_index:
            val_source.append(img)
        if int(img.split('_')[1]) in test_index:
            test_source.append(img)

        idx += 1

    random.shuffle(train_source)

    return train_source, val_source, test_source
