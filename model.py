### Load and preprocess data
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
lines = lines[1:]

#images, measurements = [], []
augmented_images, augmented_measurements = [], []
# Create adjusted steering measurements for the center, left and right camera images
correction_measurements = [0, 0.25, -0.25] # 0.25 is a constant parameter to tune

for line in lines:
    # Read in images from center, left and right cameras
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = 'data/IMG/' + filename
        imageBGR = cv2.imread(current_path)
        # opencv imread gives BGR, but drive.py reads in RGB
        image = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
        augmented_images.append(image)
        # Data Augmentation: Flipping images to help with left/right turn bias
        image_flipped = np.fliplr(image)
        # Double training dataset by adding flipped images
        augmented_images.append(image_flipped)
        
        measurement = float(line[3]) + correction_measurements[i]
        augmented_measurements.append(measurement) 
        # For flipped images, take the opposite sign of the steering measurement
        measurement_flipped = -measurement
        augmented_measurements.append(measurement_flipped)        

# Debug code, plot 10 augmented images
"""
for i in range(10):
    plt.imshow(augmented_images[i])
    plt.show()
"""

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

### Initial Setup for Keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

### Setup traning model
model = Sequential()
# Normalize and mean-center the image.
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
# The Cropping2D layer might be useful for choosing an area of interest that excludes the sky and/or the hood of the car.
# Below 4 parameters need to tune for best outcome
crop_top_row = 69
crop_bottom_row = 25
crop_left_column = 0 #60
crop_right_colum = 0 #60
model.add(Cropping2D(cropping=((crop_top_row,crop_bottom_row), (crop_left_column,crop_right_colum))))

## Build CNN (Refer to Nvidia paper)
# Conv1 - 24 filters, 5x5 kernel, 2x2 stride. Input=3@66x320, Output=24@31x158
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu',input_shape=(66, 320, 3)))

# Conv2 - 36 filters, 5x5 kernel, 2x2 stride. Output = 36@14x77
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))

# Conv3 - 48 filters, 5x5 kernel, 2x2 stride. Output = 48@5x37
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))

# Conv4 - 64 filters, 3x3 kernel, 1x1 stride. Output = 64@3x35
model.add(Convolution2D(64, 3, 3, subsample=(1,1), activation='relu'))

# Conv5 - 64 filters, 3x3 kernel, 1x1 stride. Output = 64@1x33
model.add(Convolution2D(64, 3, 3, subsample=(1,1), activation='relu'))

# Flatten
model.add(Flatten())
model.add(Dense(2112, activation='relu'))

# FC1
model.add(Dense(100, activation='relu'))

# FC2
model.add(Dense(50, activation='relu'))

# FC3: output single continous numeric value
model.add(Dense(1))


model.compile(optimizer='adam', loss='mse')
history_object = model.fit(X_train, y_train, nb_epoch=5, validation_split=0.2, shuffle=True, verbose=1)
print(history_object.history.keys())

###plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

### Save the model
model.save('model3_run7.h5')