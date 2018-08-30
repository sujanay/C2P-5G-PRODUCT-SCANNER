# Imports
import keras
import time

# importing matplotlib.pyplot caused error
import matplotlib.pyplot as plt

from keras import optimizers
from models.vgg16_model_fine_tune import vgg16_finetuned
from datagenerator.datagenerator import train_datagenerator, validation_datagenerator
from IPython.display import display
from PIL import Image

# Build vgg16 model
model = vgg16_finetuned()

# Display the model summary
print('Summary of the model')
model.summary()

# Get the datagenerator for train and validation data
print('Reading training and validation images...')
train_generator = train_datagenerator()
validation_generator = validation_datagenerator()

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# Train the model
train_start = time.clock()

print('Started training...')
history = model.fit_generator(train_generator,
                              steps_per_epoch=2*train_generator.samples/train_generator.batch_size,
                              epochs=2,
                              validation_data=validation_generator,
                              validation_steps=validation_generator.samples/validation_generator.batch_size,
                              verbose=1)

train_finish = time.clock()
train_time = train_finish - train_start

# print the time taken to train the model
print('Training completed in {0:.3f} minutes!'.format(train_time/60))

# Save the model
print('Saving the trained model...')
model.save('trained_models/model1.h5')

print("Saved trained model in 'traned_models/ folder'!")

##################################################################
############## Plot the accuracy and loss curves #################
##################################################################

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
##################################################################