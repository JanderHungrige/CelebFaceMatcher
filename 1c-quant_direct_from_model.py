#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 22:36:53 2020

@author: base
"""
#from keras_vggface.vggface import VGGFace
#from keras_vggface.utils import preprocess_input

from keras_vggface_TF.vggfaceTF import VGGFace
from keras_vggface_TF.utils import preprocess_input
print('using tf.keras')

import tensorflow as tf
tfVersion=tf.version.VERSION.replace(".", "")# use for later savename
print(tf.version.VERSION)

import cv2
import numpy as np
import pathlib as Path


#pretrained_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')  # pooling: None, avg or max


#pretrained_model.save('./1140_model.h5')

#tf>2
#converter = tf.lite.TFLiteConverter.from_keras_model(pretrained_model)

#or
converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file('./1140_model.h5')

#tf<2
#converter = tf.lite.TFLiteConverter.from_keras_model_file('./1140_model.h5')


folderpath='/home/base/Documents/Git/Projekte/CelebFaceMatcher/All_croped_images/'
size=(224,224)

def prepare(img):
    img = np.expand_dims(img,0).astype(np.float32)
    img = preprocess_input(img, version=2)
    return img
      
repDatagen=tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=prepare)
datagen=repDatagen.flow_from_directory(folderpath,target_size=size,batch_size=1)

def representative_dataset_gen():
  for _ in range(10):
    img = datagen.next()
    yield [img[0]]
    #yield [np.array(img[0], dtype=np.float32)]






# repDatagen=tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=prepare,data_format="channels_last",dtype=np.float32)
# datagen=repDatagen.flow_from_directory(folderpath,target_size=size,batch_size=1,class_mode='categorical')

# def representative_dataset_gen():
#   for _ in range(10):
#     img = datagen.next()
#     yield [img[0]]
#     #yield [np.array(img[0], dtype=np.float32)]
    
    
    
converter.representative_dataset = representative_dataset_gen

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.experimental_new_converter = True

converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.int8 
converter.inference_output_type = tf.int8 
quantized_tflite_model = converter.convert()

pather='/home/base/Documents/Git/Projekte/Quantization/'
filename=  pather + tfVersion + '_1140_quant_model.tflite' 
open(filename, "wb").write(quantized_tflite_model)
