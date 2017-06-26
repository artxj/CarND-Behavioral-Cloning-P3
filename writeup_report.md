**Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals of this project are the following:
* Use the simulator to collect data of good driving behavior on track 1 (lake) and 2 (jungle);
* Build a convolution neural network using Keras+Tensorflow that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around tracks one and two without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[flip_before]: ./examples/flip_before.png
[flip_after]: ./examples/flip_after.png
[camera_center]: ./examples/camera_center.png
[camera_left]: ./examples/camera_left.png
[camera_right]: ./examples/camera_right.png
[hist]: ./examples/hist.png
[edge1]: ./examples/edge1.png
[edge2]: ./examples/edge2.png
[edge3]: ./examples/edge3.png
[center]: ./examples/center.png
[clahe_before]: ./examples/clahe_before.png
[clahe_after]: ./examples/clahe_after.png

## Rubric Points

---
**Files Submitted & Code Quality**

**1. Submission includes all required files and can be used to run the simulator in autonomous mode**

The project includes the following files:
* [model.py](./model.py) containing the script to create and train the model;
* [drive.py](./drive.py) for driving the car in autonomous mode;
* [preprocess.py](./preprocess.py) containing the script to pre-process the image for being passed into neural network;
* [model.h5](./model.h5) containing a trained convolution neural network;
* [writeup_report.md](./writeup_report.md) summarizing the results;

Videos of the neural network successfully driving a car on both tracks are:
* [first track (lake)](./tracks/track1.mp4);
* [second track (jungle)](./tracks/track2.mp4).

**2. Submission includes functional code**
Using the Udacity provided simulator and project's drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

**3. Submission code is usable and readable**

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
The preprocess.py contains the code for pre-processing the image before passing to neural network. It is used both in model.py for training/validating the model and in drive.py for preprocessing the video input images.

## Model Architecture

**1. Model architecture**

The model is based on [NVIDIA one](https://arxiv.org/abs/1604.07316) with some changes. The model consists of 5 convolution layers with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 116-120) followed by 3 fully connected layers and output (lines 122-127). The model includes RELU activation functions after convolution layers to introduce nonlinearity.

The data is resized and normalized in the model using a Keras lambda layer (lines 106-115).

**2. Reducing overfitting in the model**

The model contains two dropout layers after fully connected ones in order to reduce overfitting (model.py lines 123-125).

The model was trained and validated on different data collected from both tracks to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the tracks.

**3. Model parameter tuning**

The model uses an adam optimizer (code line 264).

**4. Training data**

Training data was chosen to keep the vehicle driving on the road. The combination of center lane driving as well as recovering from the left and right sides of the road were used.

Details about the training data are provided in the next section

## Training Strategy

**1. Solution Design Approach**

The first step in deriving a model was to apply a NVIDIA end-to-end learning model. During the testing the changes were applied to the model.

In order to reduce the non-relevant noise from the input images they were cropped both from top and bottom. To perform calculations faster without lack in result, the images then were resized to 47x200. For normalization the standard technique of converting pixel color range from [0, 255] to [0, 1] was used.

However, the resizing resulted in removing of last convolution layer since image becomes too small. But during the tests it turned out the mean squared error on the training set was still high so the layer stays but with padding changed to 'SAME'.

Also, the first approach resulted in fast model overfitting so two dropout layers were added to the model. Surprisingly, adding ReLU activation functions after fully connected layers didn't make any improvements at all.

At the end of the process, the vehicle is able to drive autonomously around the tracks without leaving the road.

**2. Final Model Architecture**

The final model architecture (model.py lines 94-128) consists of a convolution neural network with the following layers and layer sizes:
* Conv2D (24 5x5 filters, 2x2 strides, valid padding, ReLU activation)
* Conv2D (36 5x5 filters, 2x2 strides, valid padding, ReLU activation)
* Conv2D (48 5x5 filters, 2x2 strides, valid padding, ReLU activation)
* Conv2D (64 3x3 filters, 1x1 strides, valid padding, ReLU activation)
* Conv2D (64 3x3 filters, 1x1 strides, same padding, ReLU activation)
* Fully connected layer of 100 neurons
* Dropout with probability = 0.5
* Fully connected layer of 50 neurons
* Dropout with probability = 0.5
* Fully connected layer of 10 neurons
* Output of 1 neuron


**3. Training Set & Training Process**

The process was started with capturing two laps on tracks in one direction and two in another one. The example of the captured image is:

![][center]

Then few laps were recorded with the vehicle driving back from the left and from the right sides of the road back to the center. So it prevents the vehicle from falling off the road if it gets close to the edge. The example images are:

![][edge1]
![][edge2]
![][edge3]

I started the process with the second track since it's the most complicated one. Then it was repeated for the first track to get more data and to train the network for both tracks.

Also images from left and right cameras were used with steering angle compensation value. Empirically it turns out the value of 0.15 works good for both tracks.
Sample of center image and appropriate left and right cameras:

![][camera_center]
![][camera_left]
![][camera_right]

To augment the dataset, the images were flipped with taking the opposite to the original angle. Here are the samples of the original and the flipped images:

![][flip_before]
![][flip_after]

It resulted in the following histogram of collected angles. Three peaks here indicate the most data is collected with angle = 0 (peaks to the left and to the right are for appropriate left and right cameras images):

![][hist]

To prevent falling off the track on dark areas of the second track, I have also [applied adaptive histogram equalization](http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html) to increase the contrast of the input images. Here are the samples of the original and processed images:

![][clahe_before]
![][clahe_after]

Finally the data set was randomly shuffled and 25% of the data was put into a validation set.

The training data was used in the model. I have tried different number of epochs with early stopping when validation loss stops decreasing over 2 epochs, but it turns out 4 epochs are enough for good behavior on both tracks. The Adam optimizer was used so the learning rate wasn't adjusted manually during the training.
