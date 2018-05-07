import os
import numpy as np
from keras.optimizers import SGD
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

from DeepOrg.models import resnet

from data_helpers import data_loader
from data_helpers import data_generator

batch_size = 128
nb_classes = 2
epochs = 100
data_augmentation = False

img_rows = 51
img_cols = 51
img_channels = 4
source_path = "data_source/patches"

save_result_path = 'results/%s' % ''
save_file_name = os.path.join(save_result_path, 'resnet18')

#######################################################################################
# The data, shuffled and split between train and test sets:
train, val, test = data_loader.load_source(source_path)
print('size:', len(train), len(val), len(test))
print('train:', train)
print('val', val)
print('test', test)

data_generator = data_generator.generate
loss = 'categorical_crossentropy'
optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
# initiate RMSprop optimizer
# opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# This will do preprocessing and realtime data augmentation:
print('Using real-time data augmentation.')
data_generator_train = data_generator('train', train, nb_classes, batch_size, img_channels, img_rows, img_cols, source_path)
data_generator_val = data_generator('val', val, nb_classes, batch_size, img_channels, img_rows, img_cols, source_path)

# model
model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
print(model.summary())

early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10)
csv_logger = CSVLogger(save_file_name + '.csv')
model_checkpoint = ModelCheckpoint(save_file_name + '.hdf5', monitor='val_loss', save_best_only=True)


steps_train = int(np.ceil(len(train) / float(batch_size)))
steps_val = int(np.ceil(len(val) / float(batch_size)))

#######################################################################################
print('strat training...')
history = model.fit_generator(data_generator_train, steps_per_epoch=steps_train, epochs=epochs, verbose=1,
                              validation_data=data_generator_val, validation_steps=steps_val,
                              callbacks=[csv_logger, model_checkpoint])

# print('evaluating...')
