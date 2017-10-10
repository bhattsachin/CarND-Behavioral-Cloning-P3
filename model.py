import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU
from keras.activations import relu, softmax
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
import math
import matplotlib.image as mpimg

#DEFINE FLAGS VARIABLES#
flags = tf.app.flags
FLAGS = flags.FLAGS 
flags.DEFINE_float('steering_adjustment', 0.27, "Adjustment angle.")
flags.DEFINE_integer('epochs', 25, "The number of epochs.")
flags.DEFINE_integer('batch_size', 128, "The batch size.")

columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
data = pandas.read_csv('./dataset3/driving_log.csv', skiprows=[0], names=columns)

print('data frame size', len(data.index))

#data from different angles
center = data.center.tolist()
left = data.left.tolist()
right = data.right.tolist()

print('center', len(center))
print('value of center', center[0])

#making the path relative for this docker image to read correctly
center_new = [x[x.index('dataset'):len(x)] for x in center]
print('value of new center', center_new[0])

center = center_new


#my input has been primarily this steering angle 
# given enough frame rate it can independently capture
steering = data.steering.tolist()


# SPLIT dataset
center, steering = shuffle(center, steering)
X_train, X_valid, y_train, y_valid = train_test_split(center, steering, test_size = 0.10, random_state = 100) 


# Creating buckets for driving possiblities of a steering. Although there should be more buckets in here
X_straight, X_left, X_right = [], [], []
Y_straight, Y_left, Y_right = [], [], []
for i in y_train:
  index = y_train.index(i)
  if i > 0.15:
    X_right.append(X_train[index])
    Y_right.append(i)
  if i < -0.15:
    X_left.append(X_train[index])
    Y_left.append(i)
  else:
    X_straight.append(X_train[index])
    Y_straight.append(i)

# find sample size of each direction
print('Straight:', len(X_straight), ' Left:', len(X_left), ' Right:', len(X_right))

#Straight: 1651  Left: 308  Right: 155

# we can see this track is mostly flat with some left turns but very less right turns


# TODO: do something to fix turn distribution

X_train = X_straight + X_left + X_right
y_train = np.float32(Y_straight + Y_left + Y_right)


## PART 2: Preprocessing and Augmentation

# Flip image around vertical axis
def flip(image, angle):
  new_image = cv2.flip(image,1)
  new_angle = angle*(-1)
  return new_image, new_angle

# Crop image to remove the sky and driving deck, resize to 64x64 dimension 
def crop_resize(image):
  cropped = cv2.resize(image[60:140,:], (64,64))
  return cropped



def generator_data(batch_size):
    batch_train = np.zeros((batch_size, 64, 64, 3), dtype = np.float32)
    batch_angle = np.zeros((batch_size,), dtype = np.float32)
    while True:
        data, angle = shuffle(X_train, y_train)
        for i in range(batch_size):
          choice = int(np.random.choice(len(data),1))
          batch_train[i] = crop_resize(mpimg.imread(data[choice].strip()))
          batch_angle[i] = angle[choice]*(1+ np.random.uniform(-0.10,0.10))
          #Flip random images#
          flip_coin = random.randint(0,1)
          if flip_coin == 1:
            batch_train[i], batch_angle[i] = flip(batch_train[i], batch_angle[i])

        yield batch_train, batch_angle


def generator_valid(data, angle, batch_size):
    batch_train = np.zeros((batch_size, 64, 64, 3), dtype = np.float32)
    batch_angle = np.zeros((batch_size,), dtype = np.float32)
    while True:
      data, angle = shuffle(data,angle)
      for i in range(batch_size):
        rand = int(np.random.choice(len(data),1))
        batch_train[i] = crop_resize(mpimg.imread(data[rand].strip()))
        batch_angle[i] = angle[rand]
      yield batch_train, batch_angle


### Part 3: Training


### PART 3: TRAINING ###
def main(_):
  data_generator = generator_data(FLAGS.batch_size)
  valid_generator = generator_valid(X_valid, y_valid, FLAGS.batch_size)

# Training Architecture: inspired by NVIDIA architecture #
  input_shape = (64,64,3)
  model = Sequential()
  model.add(Lambda(lambda x: x/255 - 0.5, input_shape = input_shape))
  model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample =(2,2), W_regularizer = l2(0.001)))
  model.add(Activation('relu'))
  model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample =(2,2), W_regularizer = l2(0.001)))
  model.add(Activation('relu'))
  model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample = (2,2), W_regularizer = l2(0.001)))
  model.add(Activation('relu'))
  model.add(Convolution2D(64, 3, 3, border_mode='same', subsample = (2,2), W_regularizer = l2(0.001)))
  model.add(Activation('relu'))
  model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample = (2,2), W_regularizer = l2(0.001)))
  model.add(Activation('relu'))
  model.add(Flatten())
  model.add(Dense(80, W_regularizer = l2(0.001)))
  model.add(Dropout(0.5))
  model.add(Dense(40, W_regularizer = l2(0.001)))
  model.add(Dropout(0.5))
  model.add(Dense(16, W_regularizer = l2(0.001)))
  model.add(Dropout(0.5))
  model.add(Dense(10, W_regularizer = l2(0.001)))
  model.add(Dense(1, W_regularizer = l2(0.001)))
  adam = Adam(lr = 0.0001)
  model.compile(optimizer= adam, loss='mse', metrics=['accuracy'])
  model.summary()
  model.fit_generator(data_generator, samples_per_epoch = math.ceil(len(X_train)), nb_epoch=FLAGS.epochs, validation_data = valid_generator, nb_val_samples = len(X_valid))

  print('Done Training')

###Saving Model and Weights###
  model_json = model.to_json()
  with open("model.json", "w") as json_file:
    json_file.write(model_json)
  model.save_weights("model.h5")
  print("Saved model to disk")

if __name__ == '__main__':
  tf.app.run()
