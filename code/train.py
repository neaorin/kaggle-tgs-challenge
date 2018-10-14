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

from utils import resize_image, lovasz_loss, iou, iou_2, cov_to_class, log_metric
from models import build_unet_resnet_model, build_pretrained_model

from sklearn.model_selection import train_test_split

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

from keras.preprocessing.image import array_to_img, img_to_array, load_img#,save_img

#from imgaug import augmenters as iaa

import time
t_start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--config-file', help='JSON file containing run configuration')
parser.add_argument('--data-folder', help='Folder containing input data')
parser.add_argument('--outputs-folder', help='Folder where outputs should be saved')
args = parser.parse_args()

if not args.config_file:
    print('The --config-file parameter is missing, exiting.')
    sys.exit(1)

data_folder_path = f'{args.data_folder}/data' if args.data_folder else '../data'
output_folder_path = args.outputs_folder if args.outputs_folder else '../outputs'

os.makedirs(output_folder_path, exist_ok=True)
os.makedirs(f'{output_folder_path}/models', exist_ok=True)
os.makedirs(f'{output_folder_path}/tb_logs', exist_ok=True)

with open(args.config_file) as config_file:
    config = json.load(config_file)

version = config['version']
basic_name = config['name'] % (version)

log_metric('Model Basic Name', basic_name)

img_size_ori = 101
img_size_target = 101

upsample = lambda img: resize_image(img, img_size_target)
downsample = lambda img: resize_image(img, img_size_ori)


#aug_pipeline = iaa.Sequential([
#            iaa.Fliplr(0.5) 
#        ])

datagen = ImageDataGenerator(
    horizontal_flip=True)

train_df = pd.read_csv(f'{data_folder_path}/train.csv', index_col='id', usecols=[0])
depths_df = pd.read_csv(f'{data_folder_path}/depths.csv', index_col='id')
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]
train_df['images'] = [np.array(load_img(f'{data_folder_path}/train/images/{idx}.png', color_mode='grayscale')) / 255 for idx in tqdm(train_df.index)]
train_df['masks'] = [np.array(load_img(f'{data_folder_path}/train/masks/{idx}.png', color_mode='grayscale')) / 255 for idx in tqdm(train_df.index)]
train_df['coverage'] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)     
train_df['coverage_class'] = train_df.coverage.map(cov_to_class)


# Create train/validation split stratified by salt coverage
ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
    train_df.index.values,
    np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    train_df.coverage.values,
    train_df.z.values,
    test_size=0.2, stratify=train_df.coverage_class, random_state=5678)

########################################
####                                ####
####            STAGE 1             ####
####                                ####
########################################

print('Stage 1 training.')

# model
model1 = None

input_layer = Input((img_size_target, img_size_target, 1))
output_layer = build_unet_resnet_model(input_layer, 16, 0.5)
model1 = Model(input_layer, output_layer)

c = optimizers.adam(lr = 0.01)
model1.compile(loss='binary_crossentropy', optimizer=c, metrics=[iou])

epochs = config['stage1']['epochs']
batch_size = config['stage1']['batch_size']

#early_stopping = EarlyStopping(monitor='val_iou', mode = 'max',patience=10, verbose=1)
model_checkpoint = ModelCheckpoint(f'{output_folder_path}/models/{basic_name}-stage1.model', monitor='val_iou', 
                                   mode = 'max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_iou', mode = 'max',factor=0.5, patience=5, min_lr=0.0001, verbose=1)
tb = TensorBoard(log_dir=f'{output_folder_path}/tb_logs/{basic_name}-stage1', batch_size=batch_size)

steps_per_epoch = x_train.shape[0] / batch_size

history = model1.fit_generator(datagen.flow(x_train, y_train, batch_size),
                    validation_data=[x_valid, y_valid], 
                    epochs=epochs, steps_per_epoch = steps_per_epoch,
                    callbacks=[ model_checkpoint,reduce_lr, tb], 
                    verbose=2)

print('Stage 1 training complete.')

fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15,5))

ax_loss.plot(history.epoch, history.history['loss'], label='S1 Train loss'),
ax_loss.plot(history.epoch, history.history['val_loss'], label='S1 Validation loss'),
ax_acc.plot(history.epoch, history.history['iou'], label='S1 Train IoU'),
ax_acc.plot(history.epoch, history.history['val_iou'], label='S1 Validation IoU')

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

model_stage1 = load_model(f'{output_folder_path}/models/{basic_name}-stage1.model',custom_objects={'iou': iou})

# remove activation layer and use losvasz loss
input_x = model_stage1.layers[0].input

output_layer = model_stage1.layers[-1].input
model2 = Model(input_x, output_layer)
c = optimizers.adam(lr = 0.005)

# lovasz_loss needs input range (-inf, +inf), so cancel the last "sigmoid" activation  
# Then the default threshod for pixel prediction is 0 instead of 0.5, as in my_iou_metric_2.
model2.compile(loss=lovasz_loss, optimizer=c, metrics=[iou_2])


epochs = config['stage2']['epochs']
batch_size = config['stage2']['batch_size']

early_stopping = EarlyStopping(monitor='val_iou_2', mode = 'max', patience=16, verbose=1)
model_checkpoint = ModelCheckpoint(f'{output_folder_path}/models/{basic_name}-stage2.model', monitor='val_iou_2', 
                                   mode = 'max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_iou_2', mode = 'max',factor=0.5, patience=7, min_lr=0.00001, verbose=1)
tb = TensorBoard(log_dir=f'{output_folder_path}/tb_logs/{basic_name}-stage2', batch_size=batch_size)


steps_per_epoch = x_train.shape[0] / batch_size


history = model2.fit_generator(datagen.flow(x_train, y_train, batch_size),
                    validation_data=[x_valid, y_valid], 
                    epochs=epochs, steps_per_epoch = steps_per_epoch,
                    callbacks=[ model_checkpoint,reduce_lr, tb], 
                    verbose=2)

print('Stage 2 training complete.')

fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15,5))

ax_loss.plot(history.epoch, history.history['loss'], label='S2 Train loss'),
ax_loss.plot(history.epoch, history.history['val_loss'], label='S2 Validation loss'),
ax_acc.plot(history.epoch, history.history['iou_2'], label='S2 Train IoU'),
ax_acc.plot(history.epoch, history.history['val_iou_2'], label='S2 Validation IoU')

log_metric('S2 Train loss', min(history.history['loss']))
log_metric('S2 Validation loss', min(history.history['val_loss']))
log_metric('S2 Train IoU', max(history.history['iou_2']))
log_metric('S2 Validation IoU', max(history.history['val_iou_2']))

t_finish = time.time()
print('Kernel run time = {0:.2f} minutes.'.format((t_finish-t_start)/60)) 