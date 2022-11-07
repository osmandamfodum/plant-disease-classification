from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import warnings
warnings.filterwarnings("ignore")
classifier = Sequential()

classifier.add(Convolution2D(64,3,3, input_shape=(64,64,3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(3,3)))

classifier.add(Convolution2D(64,3,3, activation='relu'))

classifier.add(MaxPooling2D(pool_size=(3,3)))

classifier.add(Flatten())




classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))



classifier.compile(optimizer ='adam', loss='binary_crossentropy', metrics=['accuracy'])


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.3,
                                    zoom_range=0.3,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/testing_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')


classifier.fit_generator(
        training_set,
        steps_per_epoch=20,
        nb_epoch=30,
        validation_data=test_set,
        nb_val_samples=10)

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/Pepper__bell___Bacterial_spot.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print (training_set.class_indices)
if result[0][0] == 1:
    prediction = 'Pepper__bell___healthy'
else:
     prediction = 'Pepper__bell___Bacterial_spot' 


print("The Prediction Result Is :",prediction)
