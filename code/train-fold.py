import os
import sys
import random

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

import json
import argparse

from utils import resize_image, lovasz_loss, iou, iou_2, iou_metric_batch, cov_to_class, log_metric, rle_encode, predict_result, focal_loss
from models import build_unet_resnet_model, build_pretrained_model, unfreeze_weights, preprocess_input

from sklearn.model_selection import train_test_split, StratifiedKFold

from tqdm import tqdm

from keras.models import Model, load_model, save_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras import optimizers

import tensorflow as tf

from keras.preprocessing.image import array_to_img, img_to_array, load_img

import time
t_start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--config-file', help='JSON file containing run configuration')
parser.add_argument('--cv-currentfold', help='The number of the specific cross-validation fold to use in this training run')
parser.add_argument('--data-folder', help='Folder containing input data')
parser.add_argument('--outputs-folder', help='Folder where outputs should be saved')
args = parser.parse_args()

if not args.config_file:
    print('The --config-file parameter is missing, exiting.')
    sys.exit(1)

data_folder_path = f'{args.data_folder}/data' if args.data_folder else '../data'
output_folder_path = args.outputs_folder if args.outputs_folder else '../outputs'
cv_currentfold = int(args.cv_currentfold) if args.cv_currentfold else None

os.makedirs(output_folder_path, exist_ok=True)
os.makedirs(f'{output_folder_path}/models', exist_ok=True)
os.makedirs(f'{output_folder_path}/figures', exist_ok=True)
os.makedirs(f'{output_folder_path}/tb_logs', exist_ok=True)
os.makedirs(f'{output_folder_path}/submissions', exist_ok=True)

with open(args.config_file) as config_file:
    config = json.load(config_file)

cv_folds = int(config['crossvalidation']['folds']) if 'crossvalidation' in config else None
version = config['version']
basic_name = config['name'] % (version)
if cv_currentfold is not None:
    basic_name = f'{basic_name}-cvfold{cv_currentfold}'

log_metric('Model Basic Name', basic_name)
log_metric('CV Current Fold', cv_currentfold)
log_metric('CV Total Folds', cv_folds)


img_size_ori = 101
img_size_target = int(config['input']['size'])
img_channels = int(config['input']['channels'])
img_normalize = bool(config['input']['normalize'])
img_normalize_divide = 255 if img_normalize else 1
img_color_mode = 'rgb' if img_channels == 3 else 'grayscale'

upsample = lambda img: resize_image(img, img_size_target)
downsample = lambda img: resize_image(img, img_size_ori)

train_df = pd.read_csv(f'{data_folder_path}/train.csv', index_col='id', usecols=[0])
depths_df = pd.read_csv(f'{data_folder_path}/depths.csv', index_col='id')
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]
train_df['images'] = [np.array(load_img(f'{data_folder_path}/train/images/{idx}.png', color_mode=img_color_mode)) / img_normalize_divide for idx in tqdm(train_df.index)]
train_df['masks'] = [np.array(load_img(f'{data_folder_path}/train/masks/{idx}.png', color_mode='grayscale')) / 255 for idx in tqdm(train_df.index)]
train_df['coverage'] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)     
train_df['coverage_class'] = train_df.coverage.map(cov_to_class)

def create_train_validation_split(index, stratify, cv_folds, cv_currentfold):
    if cv_folds is None or cv_folds == 1:
        return train_test_split(
            index.values,
            test_size=0.2, stratify=stratify, random_state=5678)
    else:
        skf = StratifiedKFold(n_splits=cv_folds, random_state=5678)
        folds = list(skf.split(np.zeros(len(train_df.index)), train_df.coverage_class))
        return index.values[folds[cv_currentfold][0]], \
            index.values[folds[cv_currentfold][1]]
  

# Create train/validation split stratified by salt coverage
idx_train, idx_valid = create_train_validation_split(train_df.index, train_df.coverage_class, cv_folds, cv_currentfold)
train_df1 = train_df.loc[idx_train]
valid_df1 = train_df.loc[idx_valid]

x_train = np.array(train_df1.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, img_channels) 
y_train = np.array(train_df1.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1) 

x_valid = np.array(valid_df1.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, img_channels) 
y_valid = np.array(valid_df1.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)

# Data augmentation
x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

data_gen_args = dict(
                    horizontal_flip=True,
                    rotation_range=20,
                    shear_range=0.3, 
                    zoom_range=0.25,
                    fill_mode='reflect'
                     )
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

datagen_seed = 1234

#image_datagen.fit(x_train, augment=True, seed=datagen_seed)
#mask_datagen.fit(y_train, augment=True, seed=datagen_seed)

image_generator = image_datagen.flow(x_train, seed=datagen_seed, batch_size=16, shuffle=True)
mask_generator = mask_datagen.flow(y_train, seed=datagen_seed, batch_size=16, shuffle=True)

# Just zip the two generators to get a generator that provides augmented images and masks at the same time
train_generator = zip(image_generator, mask_generator)

custom_objects = {
    'iou': iou, 
    'iou_2': iou_2,
    'focal_loss': focal_loss,
    'lovasz_loss': lovasz_loss
}

########################################
####                                ####
####            STAGE 1             ####
####                                ####
########################################

print('Stage 1 training.')

epochs = config['stage1']['epochs']
batch_size = config['stage1']['batch_size']
epochs_weights_frozen = int(config['stage1']['epochs_weights_frozen']) if 'epochs_weights_frozen' in config['stage1'] else 0
encoder = config['encoder'] if 'encoder' in config else None

# model
model1 = None

# start from a trained model
'''
stored_trained_model = '../outputs/models/unet-resnet-datagen-4-cvfold3-stage2.model'
print(f'Using stored trained model: {stored_trained_model}')
model1s = load_model(stored_trained_model,custom_objects=custom_objects)
input_x = model1s.layers[0].input

output_layer = Activation('sigmoid',name='output_activation')(model1s.layers[-1].output)
model1 = Model(input_x, output_layer)
'''

if encoder is not None:
    # use an architecture with an existing backbone
    model1 = build_pretrained_model(encoder['type'], encoder['backbone'], encoder['weights'], epochs_weights_frozen > 0)

    # preprocess the input for the specific backbone
    x_train = preprocess_input(x_train, encoder['backbone'])
    x_valid = preprocess_input(x_valid, encoder['backbone'])
else:
    # use a custom unet with resnet blocks
    input_layer = Input((img_size_target, img_size_target, img_channels))
    output_layer = build_unet_resnet_model(input_layer, 16, 0.5)
    model1 = Model(input_layer, output_layer)


o1 = optimizers.SGD(lr = 0.01, momentum= 0.9, decay=0.0001)
model1.compile(loss=focal_loss, optimizer=o1, metrics=[iou])

#early_stopping = EarlyStopping(monitor='val_iou', mode = 'max', patience=30, verbose=1)
model_checkpoint = ModelCheckpoint(f'{output_folder_path}/models/{basic_name}-stage1.model', monitor='iou', 
                                   mode = 'max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_iou', mode = 'max',factor=0.5, patience=20, min_lr=0.0001, verbose=1)
tb = TensorBoard(log_dir=f'{output_folder_path}/tb_logs/{basic_name}-stage1', batch_size=batch_size)

steps_per_epoch = x_train.shape[0] / batch_size

history = model1.fit_generator(train_generator,
#history = model1.fit(x_train, y_train,
                    validation_data=[x_valid, y_valid], 
                    epochs=epochs-epochs_weights_frozen, 
                    #batch_size= batch_size, 
                    steps_per_epoch = steps_per_epoch,
                    callbacks=[ model_checkpoint,reduce_lr, tb], 
                    verbose=2)

if epochs_weights_frozen > 0:
    # unfreeze weights and keep training
    unfreeze_weights(model1)
    history = model1.fit_generator(train_generator,
    #history = model1.fit(x_train, y_train,
                    validation_data=[x_valid, y_valid], 
                    initial_epoch = epochs-epochs_weights_frozen,
                    epochs=epochs, 
                    #batch_size= batch_size, 
                    steps_per_epoch = steps_per_epoch,
                    callbacks=[ model_checkpoint,reduce_lr, tb], 
                    verbose=2)

print('Stage 1 training complete.')

fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15,5))

ax_loss.plot(history.epoch, history.history['loss'], label='S1 Train loss'),
ax_loss.plot(history.epoch, history.history['val_loss'], label='S1 Validation loss'),
ax_loss.legend()
ax_acc.plot(history.epoch, history.history['iou'], label='S1 Train IoU'),
ax_acc.plot(history.epoch, history.history['val_iou'], label='S1 Validation IoU')
ax_acc.legend()

plt.savefig(f'{output_folder_path}/figures/{basic_name}-stage1-metrics.png')

log_metric('S1 Train loss', min(history.history['loss']))
log_metric('S1 Validation loss', min(history.history['val_loss']))
log_metric('S1 Train IoU', max(history.history['iou']))
log_metric('S1 Validation IoU', max(history.history['val_iou']))

########################################
####                                ####
####            STAGE 2             ####
####                                ####
########################################

print('Stage 2 training.')

model_stage1 = load_model(f'{output_folder_path}/models/{basic_name}-stage1.model',custom_objects=custom_objects)

# remove activation layer and use losvasz loss
input_x = model_stage1.layers[0].input

output_layer = model_stage1.layers[-1].input
model2 = Model(input_x, output_layer)
o2 = optimizers.SGD(lr = 0.005, momentum= 0.9, decay=0.0001)

# lovasz_loss needs input range (-inf, +inf), so cancel the last "sigmoid" activation  
# Then the default threshod for pixel prediction is 0 instead of 0.5, as in my_iou_metric_2.
model2.compile(loss=lovasz_loss, optimizer=o2, metrics=[iou_2])


epochs = config['stage2']['epochs']
batch_size = config['stage2']['batch_size']

early_stopping = EarlyStopping(monitor='val_iou_2', mode = 'max', patience=16, verbose=1)
model_checkpoint = ModelCheckpoint(f'{output_folder_path}/models/{basic_name}-stage2.model', monitor='val_iou_2', 
                                   mode = 'max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_iou_2', mode = 'max',factor=0.5, patience=8, min_lr=0.00001, verbose=1)
tb = TensorBoard(log_dir=f'{output_folder_path}/tb_logs/{basic_name}-stage2', batch_size=batch_size)


steps_per_epoch = x_train.shape[0] / batch_size


history = model2.fit_generator(train_generator,
#history = model2.fit(x_train, y_train,
                    validation_data=[x_valid, y_valid], 
                    epochs=epochs, 
                    #batch_size= batch_size, 
                    steps_per_epoch = steps_per_epoch,
                    callbacks=[ model_checkpoint,reduce_lr, tb], 
                    verbose=2)

print('Stage 2 training complete.')

fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15,5))

ax_loss.plot(history.epoch, history.history['loss'], label='S2 Train loss'),
ax_loss.plot(history.epoch, history.history['val_loss'], label='S2 Validation loss'),
ax_loss.legend()
ax_acc.plot(history.epoch, history.history['iou_2'], label='S2 Train IoU'),
ax_acc.plot(history.epoch, history.history['val_iou_2'], label='S2 Validation IoU')
ax_acc.legend()

plt.savefig(f'{output_folder_path}/figures/{basic_name}-stage2-metrics.png')

log_metric('S2 Train loss', min(history.history['loss']))
log_metric('S2 Validation loss', min(history.history['val_loss']))
log_metric('S2 Train IoU', max(history.history['iou_2']))
log_metric('S2 Validation IoU', max(history.history['val_iou_2']))


########################################
####                                ####
####            SCORING             ####
####                                ####
########################################

model = load_model(f'{output_folder_path}/models/{basic_name}-stage2.model', 
    custom_objects=custom_objects)

preds_valid = predict_result(model, x_valid)

## Scoring for last model, choose threshold by validation data 
thresholds_ori = np.linspace(0.3, 0.7, 31)
# Reverse sigmoid function: Use code below because the  sigmoid activation was removed
thresholds = np.log(thresholds_ori/(1-thresholds_ori)) 
ious = np.array([iou_metric_batch(y_valid, preds_valid > threshold) for threshold in tqdm(thresholds)])

# instead of using default 0 as threshold, use validation data to find the best threshold.
threshold_best_index = np.argmax(ious) 
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]

log_metric('Best IoU', iou_best)
log_metric('Best Threshold', threshold_best)


plt.plot(thresholds, ious)
plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
plt.xlabel("Threshold")
plt.ylabel("IoU")
plt.title(f"Threshold vs IoU ({threshold_best}, {iou_best})")
plt.legend()

plt.savefig(f'{output_folder_path}/figures/{basic_name}-iou-threshold.png')


# predict on the test dataset
# run the prediction on multiple separate slices (too much memory needed otherwise)
preds_test = None

test_batch_indexes = np.linspace(0, test_df.shape[0], 7, dtype=np.int32)
print(f'test batch indexes: {test_batch_indexes}')
for i in range(test_batch_indexes.shape[0] - 1):
    test_df_batch = test_df.iloc[test_batch_indexes[i]:test_batch_indexes[i+1]]
    test_df_batch['images'] = [np.array(load_img(f'{data_folder_path}/test/images/{idx}.png', color_mode=img_color_mode)) / img_normalize_divide for idx in tqdm(test_df_batch.index)]
    x_test_batch = np.array(test_df_batch.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, img_channels) 
    if encoder is not None:
        # preprocess the input for the specific backbone
        x_test_batch = preprocess_input(x_test_batch, encoder['backbone'])
    preds_test_batch = predict_result(model, x_test_batch)
    if preds_test is None:
        preds_test = preds_test_batch
    else:
        preds_test = np.append(preds_test, preds_test_batch, axis=0)

print(f'Predictions shape: {preds_test.shape}')

pred_dict = {idx: rle_encode(np.round(downsample(preds_test[i]) > threshold_best)) for i, idx in enumerate(tqdm(test_df.index.values))}

sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv(f'{output_folder_path}/submissions/{basic_name}-submission.csv')

t_finish = time.time()
print('Kernel run time = {0:.2f} minutes.'.format((t_finish-t_start)/60)) 