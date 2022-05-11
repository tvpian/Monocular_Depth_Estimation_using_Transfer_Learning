# -*- coding: utf-8 -*-


'''
PROJECT: Monocular Depth Estimation using Transfer Learning

GROUP: Group 9 a

TEAM MEMBERS: Pratyaksh P. Rao, Tharun P., Kaustubh Joshi, Mukund Mitra, Abhra Roy Chowdhury

'''

# Library Import
import numpy as np
import tensorflow as tf
import argparse
from PIL import Image # Imports PIL module
from io import BytesIO
from tensorflow.keras.utils import Sequence
from skimage.transform import resize
from tensorflow.keras.applications import DenseNet169
import sklearn
import os
import matplotlib.pyplot as plt

import sys

from keras import applications
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, InputLayer, Conv2D, Activation, LeakyReLU, Concatenate, UpSampling2D

from skimage import io
from zipfile import ZipFile
from tensorflow.keras import backend as K


import keras
import random

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


parser = argparse.ArgumentParser(description='My first complete deep learning code') #Input parameters
parser.add_argument('--batch_size', type=int, default=2, help='Batch size')#The batch size of the training network
parser.add_argument('--max_depth', type=int, default=1000, help='The maximal depth value')#The max depth of the images
parser.add_argument('--data', default="nyu", type=str, help='Training dataset.')#A default train dataset
parser.add_argument('--gpus', type=int, default=1, help='The number of GPUs to use')#GPU number
parser.add_argument('--epochs', type=int, default=1, help='Number of epochs') #No. of epochs
parser.add_argument('--model', type=str, default='DenseNet169', help='Custom models: DenseNet121, DenseNet201, and ResNet50') #Custom and default model for encoder-decoder network


def _parse_function_train(filename, label):

  # Probability for applying data augmentation
  add_poison_ratio = 0.5
  color_ratio = 0.5
  mirror_ratio = 0.5
  flip_ratio = 0.5
  brightness_ratio = 0.5
  saturation_ratio = 0.5

  # Read images from disk
  shape_rgb = (480, 640, 3)
  shape_depth = (480, 640, 1)
  image_decoded = tf.image.decode_jpeg(tf.io.read_file(filename))
  depth_resized = tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(label)),
                                  [shape_depth[0], shape_depth[1]])

  # Format
  rgb = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
  depth = tf.image.convert_image_dtype(depth_resized / 255.0, dtype=tf.float32)

  # Normalize the depth values (in cm)
  depth = 1000 / tf.clip_by_value(depth * 1000, 10, 1000)

  # Data augmentation

  # Salt and Pepper Noise
  if random.uniform(0, 1) <= add_poison_ratio:
    PEAK = 20
    rgb = tf.experimental.numpy.random.poisson(tf.clip_by_value(rgb, 0, 1) * PEAK) / PEAK

  # Colour channel switching from RGB to BGR
  if random.uniform(0, 1) <= color_ratio:
    rgb = rgb[:, :, ::-1]

  # Image mirroring
  if random.uniform(0, 1) <= mirror_ratio:
    rgb = tf.image.flip_left_right(rgb)
    depth = tf.image.flip_left_right(depth)

  # Vertical flip
  if random.uniform(0, 1) <= flip_ratio:
    rgb = tf.image.random_flip_up_down(rgb, seed=None)
    depth = tf.image.flip_left_right(depth)

  # Brightness manipulation
  if random.uniform(0, 1) <= brightness_ratio:
    rgb = tf.image.random_brightness(rgb, max_delta, seed=None)

  # Saturation adjustment
  if random.uniform(0, 1) <= saturation_ratio:
    rgb = tf.image.random_saturation(rgb, lower, upper, seed=None)
    
  return rgb, depth

def _parse_function_test(filename, label):
  # Read images from disk
  shape_rgb = (480, 640, 3)
  shape_depth = (480, 640, 1)
  image_decoded = tf.image.decode_jpeg(tf.io.read_file(filename))
  depth_resized = tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(label)),
                                  [shape_depth[0], shape_depth[1]])

  # Format
  rgb = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
  depth = tf.image.convert_image_dtype(depth_resized / 255.0, dtype=tf.float32)

  # Normalize the depth values (in cm)
  depth = 1000 / tf.clip_by_value(depth * 1000, 10, 1000)

  return rgb, depth

if args.data == 'nyu':
    ## Train_dataset

    # Revise the dataset folder
    csv_file = '#Revise the dataset folder#'
    csv = open(csv_file, 'r').read()
    nyu2_train = list((row.split(',') for row in (csv).split('\n') if len(row) > 0))
    nyu2_train = sklearn.utils.shuffle(nyu2_train, random_state=0)

    # Data truncation to reduce training time ( 1000 images (1.754%) of the total training dataset)
    nyu2_train = nyu2_train[0:1000]

    # Revise the dataset folder
    filenames = [os.path.join('#Revise the dataset folder#',i[0]) for i in nyu2_train]
    labels = [os.path.join('#Revise the dataset folder#',i[1])for i in nyu2_train]
    length_train = len(filenames)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(buffer_size=len(filenames), reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.map(map_func=_parse_function_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    batch_size = args.batch_size  # batch_size from inputs, default value is 2
    train_generator = dataset.batch(batch_size=batch_size)
    ## Test_dataset
    csv_file_test = '#Revise the dataset folder#'
    csv_test = open(csv_file_test, 'r').read()
    nyu2_test = list((row.split(',') for row in (csv_test).split('\n') if len(row) > 0))
    nyu2_test = sklearn.utils.shuffle(nyu2_test, random_state=0)

    # Data truncation to reduce training time (50 images (7.645%) of the total testing dataset)
    nyu2_test = nyu2_test[0:50]

    # Revise the dataset folder
    filenames_test = [os.path.join('#Revise the dataset folder#', i[0]) for i in nyu2_test]
    labels_test = [os.path.join('#Revise the dataset folder#', i[1]) for i in nyu2_test]
    length_test = len(filenames_test)
    print(labels_test)
    dataset_test = tf.data.Dataset.from_tensor_slices((filenames_test, labels_test))
    dataset_test = dataset_test.shuffle(buffer_size=len(filenames_test), reshuffle_each_iteration=True)
    dataset_test = dataset_test.repeat()
    dataset_test = dataset_test.map(map_func=_parse_function_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    batch_size = args.batch_size  # batch_size from inputs, default value is 2
    test_generator = dataset_test.batch(batch_size=batch_size)
    print(length_test)


# Base and custom defined encoder-decoder networks (DensNet169 (base model), DenseNet121, DenseNet201, and ResNet50 respectively)
if (args.model == 'DenseNet169' or 'DenseNet121' or 'DenseNet201'):

  # Encoder

  # DenseNet169
  if args.model == 'DenseNet169':
    base_model = tf.keras.applications.densenet.DenseNet169(
        weights='imagenet', input_tensor=None,
        pooling=None, classes=1000, input_shape=(480, 640, 3), include_top=False
    )

  # DenseNet121
  elif args.model == 'DenseNet121':
    base_model = tf.keras.applications.densenet.DenseNet121(
        weights='imagenet', input_tensor=None,
        pooling=None, classes=1000, input_shape=(480, 640, 3), include_top=False
    )

  # DenseNet201
  elif args.model == 'DenseNet201':
    base_model = tf.keras.applications.densenet.DenseNet201(
        weights='imagenet', input_tensor=None,
        pooling=None, classes=1000, input_shape=(480, 640, 3), include_top=False
    )



  base_model_output_shape = base_model.layers[-1].output.shape
  for layer in base_model.layers: layer.trainable = True

  decode_filters = int(base_model_output_shape[-1])

  # Define upsampling block
  def upsampler(tensor, filters, name, concat_with):
      up_i = UpSampling2D((2, 2),interpolation="bilinear", name=name+'_upsampling2d')(tensor)
      up_i = Concatenate(name=name+'_concat')([up_i, base_model.get_layer(concat_with).output]) # Skip connection
      up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convA')(up_i)
      up_i = LeakyReLU(alpha=0.2)(up_i)
      up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convB')(up_i)
      up_i = LeakyReLU(alpha=0.2)(up_i)
      return up_i

  #### Decoder model ############
  decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same',
                   input_shape=base_model_output_shape, name='conv2')(base_model.output)

  decoder = upsampler(decoder, int(decode_filters/2), 'up1', concat_with='pool3_pool')
  decoder = upsampler(decoder, int(decode_filters/4), 'up2', concat_with='pool2_pool')
  decoder = upsampler(decoder, int(decode_filters/8), 'up3', concat_with='pool1')
  decoder = upsampler(decoder, int(decode_filters/16), 'up4', concat_with='conv1/relu')

  # Resizing to original grount truth depth resolution (480 x 640 x 1)
  outputs1_5 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(decoder)
  outputs_final=tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding='same')(outputs1_5)


# Encoder

# ResNet50
if args.model == 'ResNet50':
  print('ResNet50 model initialized')
  base_model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', 
          include_top=False, input_shape=(480, 640, 3))


  base_model_output_shape = base_model.layers[-1].output.shape
  for layer in base_model.layers: layer.trainable = True

  decode_filters = int(base_model_output_shape[-1])

  # Define upsampling block
  def upsampler(tensor, filters, name, concat_with):
      up_i = UpSampling2D((2, 2),interpolation="bilinear", name=name+'_upsampling2d')(tensor)
      up_i = Concatenate(name=name+'_concat')([up_i, base_model.get_layer(concat_with).output]) # Skip connection
      up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convA')(up_i)
      up_i = LeakyReLU(alpha=0.2)(up_i)
      up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convB')(up_i)
      up_i = LeakyReLU(alpha=0.2)(up_i)
      return up_i

  #### Decoder model ############
  decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=base_model_output_shape, name='conv2')(base_model.output)
  decoder = upsampler(decoder, int(decode_filters/2), 'up1', concat_with='conv4_block1_2_conv')
  decoder = upsampler(decoder, int(decode_filters/4), 'up2', concat_with='conv3_block1_2_conv')
  decoder = upsampler(decoder, int(decode_filters/8), 'up3', concat_with='pool1_pool')
  decoder = upsampler(decoder, int(decode_filters/16), 'up4', concat_with='conv1_relu')

  outputs1_5 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(decoder)
  outputs_final=tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding='same')(outputs1_5)


model=tf.keras.Model(inputs=base_model.inputs, outputs=outputs_final)
print('\nModel created.')

print(model.summary())
######################### Multi-gpu setup:################################

basemodel = model
if args.gpus > 1: model = tf.keras.utils.multi_gpu_model(model, gpus=args.gpus)

# Loss function
def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=1000.0 / 10.0):
    #L1 Loss
    l_depth = tf.keras.backend.mean(tf.keras.backend.abs(y_pred - y_true), axis=-1)

    #L2 Loss
    l_l2 = tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1)
    
    # Edges
    #Gradient Loss
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = tf.keras.backend.mean(tf.keras.backend.abs(dy_pred - dy_true) + tf.keras.backend.abs(dx_pred - dx_true), axis=-1)

    # Structural similarity (SSIM) index
    l_ssim = tf.keras.backend.clip((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)
   
    # Berhu Loss
  
    abs_error = tf.abs(tf.subtract(y_pred, y_true), name='abs_error')
 
    # Calculate threshold c from max error
    c = 0.2 * tf.reduce_max(abs_error)

    berHu_loss = tf.where(abs_error <= c,   
                       abs_error, 
                      (tf.square(abs_error) + tf.square(c))/(2*c))
 
    loss_berHu = tf.reduce_sum(berHu_loss)

    # Weighted loss sum

    w1 = 0.8
    w2 = 1.6
    w3 = theta
    w4 = 0.0
  
    return (w1 * l_ssim) + (w2 * tf.keras.backend.mean(l_edges)) + (w3 * tf.keras.backend.mean(l_depth)) + w4*loss_berHu

print('\n\n\n', 'Compiling model..')
######################### Trainning ################################
learning_rate=0.0001
model.compile(optimizer=tf.optimizers.Adam(1e-2,lr=learning_rate, amsgrad=True),loss=depth_loss_function)
print('\n\n\n', 'Compiling complete')


model.fit(train_generator,epochs=args.epochs,steps_per_epoch=length_train//batch_size)
###########################Save model###############################

model.save("./models/model_with.h5", include_optimizer=False)

model.save('./models/', save_format='tf',include_optimizer=False)

##########################Result test################################
score=model.evaluate(test_generator,steps=10)

print("last score:",score)

#########################Predict a result#############################
image_decoded = tf.image.decode_jpeg(tf.io.read_file('1.jpg'))
rgb = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
rgb=np.expand_dims(rgb, axis = 0)
#model = tf.keras.models.load_model('./models/model.h5',custom_objects={'depth_loss_function': depth_loss_function})
result=model.predict(rgb)
#print(result)
image_new=result[0,:,:,0]
plt.imshow(image_new)
plt.show()
