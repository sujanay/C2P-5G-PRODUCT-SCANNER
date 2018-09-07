# Imports
import keras
import time

# importing matplotlib.pyplot caused error
# import matplotlib.pyplot as plt

from keras import optimizers
from models.vgg16_model import vgg16_finetuned
from datagenerator.datagenerator import train_datagenerator, validation_datagenerator
from IPython.display import display
from PIL import Image
from keras.models import load_model

# Build vgg16 model
loaded_model = load_model('trained_models/model1.h5')

# Display the model summary
print('Summary of the model')
loaded_model.summary()

# Get the datagenerator for train and validation data
print('Reading training and validation images...')
train_generator = train_datagenerator()
validation_generator = validation_datagenerator()

# Train the model
train_start = time.clock()

print('Started training...')
history = loaded_model.fit_generator(train_generator,
                                     steps_per_epoch=2*train_generator.samples/train_generator.batch_size,
                                     epochs=2,
                                     validation_data=validation_generator,
                                     validation_steps=validation_generator.samples/validation_generator.batch_size,
                                     verbose=1)

train_finish = time.clock()
train_time = train_finish - train_start

# print the time taken to train the model
print('Training completed in {0:.3f} minutes!'.format(train_time/60))

