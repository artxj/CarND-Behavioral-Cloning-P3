
import csv
import cv2
from keras.models import Sequential, Model
from keras.layers import Conv2D, Cropping2D, Dense, Flatten, Lambda
import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def read_data(file_path):
    samples = []
    with open(file_path) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            samples.append(line)
    return samples

def flip_image(image, angle):
    return cv2.flip(image, flipCode=1), -angle

def read_image_data(file_path, angle, shift=0.0):
    image = cv2.imread(file_path)
    return image, (angle + shift)

def convert_path(path):
    path_components = path.split('/')[-4:]
    return './' + '/'.join(path_components)

def samples_generator(samples, batch_size=32, shift=0.2):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image, center_angle = read_image_data(convert_path(batch_sample[0]), float(batch_sample[3]))
                left_image, left_angle = read_image_data(convert_path(batch_sample[1]), center_angle, shift)
                right_image, right_angle = read_image_data(convert_path(batch_sample[2]), center_angle, -shift)

                if center_image is None or left_image is None or right_image is None:
                    continue

                images = images + [center_image, left_image, right_image]
                angles = angles + [center_angle, left_angle, right_angle]

                flipped_center_image, flipped_center_angle = flip_image(center_image, center_angle)
                flipped_left_image, flipped_left_angle = flip_image(left_image, left_angle)
                flipped_right_image, flipped_right_angle = flip_image(right_image, right_angle)

                images = images + [flipped_center_image, flipped_left_image, flipped_right_image]
                angles = angles + [flipped_center_angle, flipped_left_angle, flipped_right_angle]

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

def build_model(cropping, input_shape):
    model = Sequential()
    model.add(Cropping2D(cropping=cropping, input_shape=(160, 320, 3)))
    new_shape = (input_shape[0] - cropping[0][0] - cropping[0][1], input_shape[1] - cropping[1][0] - cropping[1][1], input_shape[2])
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=new_shape))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

def plot_history(history_object):
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

log_paths = [ './data/data1_1/driving_log.csv', './data/data1_2/driving_log.csv', './data/data1_3/driving_log.csv',\
    './data/data2_1/driving_log.csv', './data/data2_2/driving_log.csv' ]
model_path = './model/model.h5'
nb_epoch = 5
image_shape = (160, 320, 3)
batch_size = 32
angle_shift = 0.2

print('Loading samples...')

samples = []
for path in log_paths:
    samples = samples + read_data(path)

train_samples, validation_samples = train_test_split(samples, test_size=0.33)
nb_train_samples = len(train_samples)
nb_validation_samples = len(validation_samples)

print('loaded!')
print('Train samples count = {}'.format(nb_train_samples))
print('Validation samples count = {}'.format(nb_validation_samples))

train_generator = samples_generator(train_samples, batch_size=batch_size, shift=angle_shift)
validation_generator = samples_generator(validation_samples, batch_size=batch_size, shift=angle_shift)

model = build_model(cropping=((60, 25), (0, 0)), input_shape=image_shape)
model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size,\
    validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size,\
    epochs=nb_epoch)

plot_history(history)

model.save(model_path)
