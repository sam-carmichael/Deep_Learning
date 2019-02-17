import csv
import cv2
import numpy as np

images = []
measurements = []

with open('driving_log.csv') as csvfile:
    reader =  csv.reader(csvfile)
    for row in reader:
        steering_center = float(row[3])

        # create adjusted steering measurements for the side camera images
        # this is a parameter to tune
        correction = 0.2
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # for center's image
        source_path = row[0]
        filename = source_path.split('/')[-1]
        current_path = 'IMG/' + filename
        img_center = cv2.imread(current_path)

        # for left's image
        source_path = row[1]
        filename = source_path.split('/')[-1]
        current_path = 'IMG/' + filename
        img_left = cv2.imread(current_path)

        # for right's image
        source_path = row[2]
        filename = source_path.split('/')[-1]
        current_path = 'IMG/' + filename
        img_right = cv2.imread(current_path)

        images.extend([img_center, img_left, img_right])
        measurements.extend([steering_center, steering_left, steering_right])

X_train = np.array(images)
Y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Cropping2D
from keras.layers import Dropout

model = Sequential()
# need to use lamdab, else it will hit memory overflow.
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160, 320, 3)))
# add the Cropping Layer
model.add(Cropping2D(cropping=((75,25), (0,0))))
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.45))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
#model.fit(X_train / 255.0 - 0.5, Y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model_32th.h5')
