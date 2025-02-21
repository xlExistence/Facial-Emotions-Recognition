from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

train_data_dir = 'data_fer2013/train'
validation_data_dir = 'data_fer2013/test/'


train_datagen = ImageDataGenerator(
					rescale = 1./255,
					rotation_range = 30,
					shear_range = 0.3,
					zoom_range = 0.3,
					horizontal_flip = True,
					fill_mode = 'nearest')
validation_datagen = ImageDataGenerator(rescale = 1./255)