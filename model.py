
import argparse
import csv
import cv2
import keras
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Activation, Conv2D, Cropping2D, Dense, Dropout, Flatten, Lambda
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from scipy.misc import imresize
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from preprocess import clahe_image

def read_data(file_path):
    """
    Reads driving log and returns its lines as array
    """
    samples = []
    with open(file_path) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            samples.append(line)
    return samples

def flip_image(image, angle):
    """
    Flips image horizontally. Returns (-angle) as second parameter
    """
    return np.fliplr(image), -angle

def read_image_data(file_path, angle, shift=0.0):
    """
    Reads an image. Returns (angle+shift) as second parameter
    """
    image = cv2.imread(file_path)
    if image is None:
        return None, 0
    image = clahe_image(image)
    return image, (angle + shift)

def convert_path(path):
    """
    Converts an absolute image path to relative one
    """
    path_components = path.split('/')[-4:]
    return './' + '/'.join(path_components)

def samples_generator_v2(samples, batch_size=32, shift=0.2, flip=False):
    """
    Generator to produce batches of samples images.
    Produces 3 x batch_size images if flip=False (center, left, right)
    Produces 6 x batch_size images if flip=True (center, left, right, all flipped ones)
    """
    num_samples = len(samples)

    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                center_angle = float(batch_sample[3])
                for image_num in range(0, 3):
                    apply_shift = 0.0
                    if image_num == 1:
                        apply_shift = shift
                    elif image_num == 2:
                        apply_shift = -shift
                    orig_image, orig_angle = read_image_data(convert_path(batch_sample[image_num]), center_angle, apply_shift)
                    if orig_image is None:
                        continue
                    images.append(orig_image)
                    angles.append(orig_angle)
                    if flip:
                        flipped_image, flipped_angle = flip_image(orig_image, orig_angle)
                        images.append(flipped_image)
                        angles.append(flipped_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            # y_train = np.array((angles, speeds)).T
            yield shuffle(X_train, y_train)

def build_model(cropping, input_shape):
    """
    Builds Keras model.
    NVIDIA model https://arxiv.org/abs/1604.07316 with small changes:
    the 5th conv layer has same padding due to lower image size;
    two dropout layers are added after FCs.
    """
    model = Sequential()

    model.add(Cropping2D(cropping=cropping, input_shape=(160, 320, 3)))
    new_shape = (input_shape[0] - cropping[0][0] - cropping[0][1], input_shape[1] - cropping[1][0] - cropping[1][1], input_shape[2])

    def resize_normalize_image(image):
        """
        Resizing the input image to (47, 200)
        and normalizing it.
        """
        import tensorflow as tf
        image = tf.image.resize_images(image, (47, 200))
        return (image / 255.0) - 0.5

    model.add(Lambda(resize_normalize_image, input_shape=new_shape, output_shape=(47, 200, 3)))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

def plot_history(history_object):
    """
    Displays the history plot of MSE loss over epochs
    """
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

def show_histogram(samples, shift=0.2):
    """
    Displays histogram on the distribution of the data provided
    """
    y_data = np.array([])
    for sample in samples:
        angle = float(sample[3])
        y_data = np.append(y_data, angle)
        y_data = np.append(y_data, angle+shift)
        y_data = np.append(y_data, angle-shift)

    plt.hist(y_data, bins=51)
    plt.show()

def test_clahe():
    '''
    Displaying the test image after applying CLAHE.
    '''
    img = cv2.imread('./mouse_data/data2_1/IMG/center_2017_06_19_18_46_53_190.jpg')

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    plt.imshow(final)
    plt.show()

def parse_args():
    '''
    Parses console arguments.
    '''
    parser = argparse.ArgumentParser(description='Car steering angle prediction model.')
    parser.add_argument(
        '--model_path',
        type=str,
        default='./model/model.h5',
        help='Path to a model to be loaded (optionally) and to be saved'
    )
    parser.add_argument(
        '--load',
        action='store_true',
        help='Load model from model_path.'
    )
    parser.add_argument(
        '--hist',
        action='store_true',
        help='Display a histogram of input data.'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Displays a plot history of val_loss.'
    )
    parser.add_argument(
        '--test_clahe',
        action='store_true',
        help='Displays a sample of applying CLAHE for image.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=15,
        help='Number of epochs to be run.'
    )
    return parser.parse_args()

def main():
    """
    Creates/Loads the model and runs training over it.
    """

    args = parse_args()

    if args.test_clahe:
        test_clahe()
        return

    log_paths = [ './mouse_data/data1_1/driving_log.csv',
        './mouse_data/data1_2/driving_log.csv',
        './mouse_data/data1_3/driving_log.csv',
        './mouse_data/data1_4/driving_log.csv',
        './mouse_data/data2_1/driving_log.csv',
        './mouse_data/data2_2/driving_log.csv',
        './mouse_data/data2_3/driving_log.csv',
        './mouse_data/data2_4/driving_log.csv' ]

    model_path = args.model_path
    nb_epoch = args.epochs
    image_shape = (160, 320, 3)
    batch_size = 128
    angle_shift = 0.15

    print('Loading samples...')

    samples = []
    for path in log_paths:
        samples = samples + read_data(path)

    train_samples, validation_samples = train_test_split(samples, test_size=0.25)
    nb_train_samples = len(train_samples)
    nb_validation_samples = len(validation_samples)

    if args.hist:
        show_histogram(train_samples, angle_shift)
        return

    print('...loaded!')
    print('Train samples count = {}'.format(nb_train_samples))
    print('Validation samples count = {}'.format(nb_validation_samples))

    train_generator = samples_generator_v2(train_samples, batch_size=batch_size,
        shift=angle_shift, flip=True)
    validation_generator = samples_generator_v2(validation_samples, batch_size=batch_size,
        shift=angle_shift, flip=True)

    if args.load:
        model = keras.models.load_model(model_path)
    else:
        model = build_model(cropping=((60, 25), (0, 0)), input_shape=image_shape)

    model.compile(loss='mse', optimizer='adam')
    history = model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size,
        validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size,
        callbacks=[EarlyStopping(monitor='val_loss', min_delta=.0, patience=1)],
        epochs=nb_epoch)

    model.save(model_path)

    if args.plot:
        plot_history(history)


if __name__ == '__main__':
    main()
