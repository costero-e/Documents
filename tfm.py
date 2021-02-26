# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import glob
import os
import pathlib

from numpy import loadtxt
from keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import sys, getopt

def main(argv):
  print("1")
  #print("argument:")
  urlpath = str(argv[0])
  # print(urlpath)
  # print("test!")
  # print(argv)
  print("2")
  # path_train = '/Users/barnatasa/Desktop/Màster Bioinformàtica/TFM/lung_colon_image_set/train'
  path_train = '/Library/WebServer/Documents/train'
  print("3")
  # tifCounter = len(glob.glob1(path_train,"*.jpeg"))
  # print(tifCounter)
  
  batch_size = 32
  img_height = 180
  img_width = 180

  print("3.15")

  train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path_train,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

  print("3.5")
  # path_val = '/Users/barnatasa/Desktop/Màster Bioinformàtica/TFM/lung_colon_image_set/validation'
  path_val = '/Library/WebServer/Documents/validation'
  val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path_val,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

  print("4")
  class_names = train_ds.class_names
  # print(class_names)

  AUTOTUNE = tf.data.AUTOTUNE

  train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
  val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
  print("5")
  normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

  # load model
  # model = load_model('/Users/barnatasa/Desktop/Màster Bioinformàtica/TFM/lung_2.h5')
  model = load_model('/Library/WebServer/Documents/lung_2.h5')

  '''
  list_of_files = glob.glob('/Library/WebServer/Documents/uploads/*') # * means all if need specific format then *.csv
  latest_file = max(list_of_files, key=os.path.getctime)
  path = os.path.abspath(latest_file)
  '''

  image_url = "https://i.ibb.co/DVF8fhD/lungaca3500.jpg"
  print("6")
  data_dir = tf.keras.utils.get_file('lung', origin=urlpath)
  # data_dir = tf.keras.utils.get_file('lung', origin=image_url)
  data_dir = pathlib.Path(data_dir)
  print("7")
  

  img = keras.preprocessing.image.load_img(
      data_dir, target_size=(img_height, img_width)
  )
  img_array = keras.preprocessing.image.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0)
  print("8")
  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])

  print(
      "This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(class_names[np.argmax(score)], 100 * np.max(score))
  )

if __name__ == "__main__":
   main(sys.argv[1:])