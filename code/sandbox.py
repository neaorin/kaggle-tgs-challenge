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

from utils import resize_image, lovasz_loss, iou, iou_2, iou_metric_batch, cov_to_class, log_metric, rle_encode, predict_result
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

#from imgaug import augmenters as iaa

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
os.makedirs(f'{output_folder_path}/tb_logs', exist_ok=True)
os.makedirs(f'{output_folder_path}/submissions', exist_ok=True)

with open(args.config_file) as config_file:
    config = json.load(config_file)

cv_folds = int(config['crossvalidation']['folds']) if config['crossvalidation'] is not None else None
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


#aug_pipeline = iaa.Sequential([
#            iaa.Fliplr(0.5) 
#        ])

datagen = ImageDataGenerator(
    horizontal_flip=True)

train_df = pd.read_csv(f'{data_folder_path}/train.csv', index_col='id', usecols=[0])
depths_df = pd.read_csv(f'{data_folder_path}/depths.csv', index_col='id')
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]
train_df['images'] = [np.array(load_img(f'{data_folder_path}/train/images/{idx}.png', color_mode=img_color_mode)) / img_normalize_divide for idx in tqdm(train_df.index)]
train_df['masks'] = [np.array(load_img(f'{data_folder_path}/train/masks/{idx}.png', color_mode='grayscale')) / 255 for idx in tqdm(train_df.index)]
train_df['coverage'] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)     
train_df['coverage_class'] = train_df.coverage.map(cov_to_class)

encoder = config['encoder']

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
########################################
####                                ####
####            SCORING             ####
####                                ####
########################################

model = load_model(f'{output_folder_path}/models/{basic_name}-stage2.model', 
    custom_objects={'iou_2': iou_2, 'lovasz_loss': lovasz_loss})

preds_valid = predict_result(model, x_valid)

iou_best = 0.3906093906093906
threshold_best =  0.2141479884056315

log_metric('Best IoU', iou_best)
log_metric('Best Threshold', threshold_best)


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