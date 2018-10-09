import os
import sys
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse

from utils import resize_image, image_generator, lovasz_loss, my_iou_metric, my_iou_metric_2, cov_to_class

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from keras.models import Model, load_model, save_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras import backend as K
from keras import optimizers

import tensorflow as tf

from keras.preprocessing.image import array_to_img, img_to_array, load_img#,save_img

from imgaug import augmenters as iaa

import time
t_start = time.time()

abspath = os.path.abspath(__file__)
dname = os.path.dirname(os.path.dirname(abspath))

parser = argparse.ArgumentParser()
parser.add_argument("--config-file", help="JSON file containing run configuration")
args = parser.parse_args()

if not args.config_file:
    print('The --config-file parameter is missing, exiting.')
    sys.exit(1)

with open(args.config_file) as config_file:
    config = json.load(config_file)

version = config['version']
basic_name = config['name'] % (version)

print(basic_name)

img_size_ori = 101
img_size_target = 101

upsample = lambda img: resize_image(img, img_size_target)
downsample = lambda img: resize_image(img, img_size_ori)


aug_pipeline = iaa.Sequential([
            iaa.Fliplr(0.5) 
        ])


train_df = pd.read_csv("{0}/data/train.csv".format(dname), index_col="id", usecols=[0])
depths_df = pd.read_csv("{0}/data/depths.csv".format(dname), index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]
train_df["images"] = [np.array(load_img("{0}/data/train/images/{1}.png".format(dname, idx), color_mode="grayscale")) / 255 for idx in tqdm(train_df.index)]
train_df["masks"] = [np.array(load_img("{0}/data/train/masks/{1}.png".format(dname, idx), color_mode="grayscale")) / 255 for idx in tqdm(train_df.index)]
train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)     
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)


# Create train/validation split stratified by salt coverage
ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
    train_df.index.values,
    np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    train_df.coverage.values,
    train_df.z.values,
    test_size=0.2, stratify=train_df.coverage_class, random_state=5678)

model_stage1 = load_model('{0}/models/{1}-stage1.model'.format(dname, basic_name) ,custom_objects={'my_iou_metric': my_iou_metric})

# remove layter activation layer and use losvasz loss
input_x = model_stage1.layers[0].input

output_layer = model_stage1.layers[-1].input
model = Model(input_x, output_layer)
c = optimizers.adam(lr = 0.01)

# lovasz_loss needs input range (-inf, +inf), so cancel the last "sigmoid" activation  
# Then the default threshod for pixel prediction is 0 instead of 0.5, as in my_iou_metric_2.
model.compile(loss=lovasz_loss, optimizer=c, metrics=[my_iou_metric_2])


epochs = config['stage2']['epochs']
batch_size = config['stage2']['batch_size']

early_stopping = EarlyStopping(monitor='val_my_iou_metric_2', mode = 'max',patience=20, verbose=1)
model_checkpoint = ModelCheckpoint('{0}/models/{1}-stage2.model'.format(dname, basic_name), monitor='my_iou_metric_2', 
                                   mode = 'max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='my_iou_metric_2', mode = 'max',factor=0.5, patience=5, min_lr=0.0001, verbose=1)
tb = TensorBoard(log_dir="{0}/tb_logs/{1}".format(dname, basic_name), batch_size=batch_size)


steps_per_epoch = x_train.shape[0] / batch_size


history = model.fit_generator(generator(x_train, y_train, batch_size, aug_pipeline),
                    validation_data=[x_valid, y_valid], 
                    epochs=epochs, steps_per_epoch = steps_per_epoch,
                    initial_epoch = continue_from_epoch,
                    callbacks=[ model_checkpoint,reduce_lr,early_stopping, tb], 
                    verbose=2)

t_finish = time.time()
print("Kernel run time = {0:.2f} minutes.".format((t_finish-t_start)/60)) 
