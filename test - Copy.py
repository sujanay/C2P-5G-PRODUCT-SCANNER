<<<<<<< HEAD
# Imports
import cv2
import keras
import time
import numpy as np
from keras import optimizers
from models.vgg16_model_fine_tune import vgg16_finetuned
from datagenerator.datagenerator import test_datagenerator
from keras.preprocessing.image import load_img
from IPython.display import display
from PIL import Image
from keras.models import load_model

test_dir = 'dataset/test'

# Load the trained model
print('loading trained model...')
new_model = load_model('trained_models/model1.h5')
print('loading complete')

# Print the summary of the loaded model
print('summary of loaded model')
new_model.summary()

# Load test images
print('reading test images...')
test_generator = test_datagenerator()

# Get the filenames from the generator
fnames = test_generator.filenames

# Get the ground truth from generator
ground_truth = test_generator.classes

# Get the label to class mapping from the generator
label2index = test_generator.class_indices

# Getting the mapping from class index to class label
idx2label = dict((v, k) for k, v in label2index.items())

# Get the predictions from the model using the generator
print('predicting on the test images...')

prediction_start = time.clock()
predictions = new_model.predict_generator(test_generator,
                                          steps=test_generator.samples / test_generator.batch_size,
                                          verbose=0)

prediction_finish = time.clock()
prediction_time = prediction_finish - prediction_start

predicted_classes = np.argmax(predictions, axis=1)

errors = np.where(predicted_classes != ground_truth)[0]
print("No. of errors = {}/{}".format(len(errors), test_generator.samples))

correct_predictions = np.where(predicted_classes == ground_truth)[0]
print("No. of correct predictions = {}/{}".format(len(correct_predictions), test_generator.samples))

print("Test Accuracy = {0:.2f}%".format(len(correct_predictions)*100/test_generator.samples))

print("Predicted in {0:.3f} minutes!".format(prediction_time/60))

# original = load_img('{}/{}'.format(test_dir, fnames[correct_predictions[2]]))
# print(original)
# # Show the errors
# for i in range(len(errors)):
#     pred_class = np.argmax(predictions[errors[i]])
#     pred_label = idx2label[pred_class]
#
#     title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
#         fnames[errors[i]].split('/')[0],
#         pred_label,
#         predictions[errors[i]][pred_class])
#
#     original = load_img('{}/{}'.format(test_dir, fnames[errors[i]]))
#     plt.figure(figsize=[7, 7])
#     plt.axis('off')
#     plt.title(title)
#     plt.imshow(original)
#     plt.show()
#
# # Show the correct predictions
# for i in range(len(correct_predictions)):
#     pred_class = np.argmax(predictions[correct_predictions[i]])
#     pred_label = idx2label[pred_class]
#
#     title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
#         fnames[correct_predictions[i]].split('/')[0],
#         pred_label,
#         predictions[correct_predictions[i]][pred_class])
#
#     original = load_img('{}/{}'.format(test_dir, fnames[correct_predictions[i]]))
#     plt.figure(figsize=[7, 7])
#     plt.axis('off')
#     plt.title(title)
#     plt.imshow(original)
=======
# Imports
import cv2
import keras
import time
import numpy as np
from keras import optimizers
from models.vgg16_model_fine_tune import vgg16_finetuned
from datagenerator.datagenerator import test_datagenerator
from keras.preprocessing.image import load_img
from IPython.display import display
from PIL import Image
from keras.models import load_model

test_dir = 'dataset/test'

# Load the trained model
print('loading trained model...')
new_model = load_model('trained_models/model1.h5')
print('loading complete')

# Print the summary of the loaded model
print('summary of loaded model')
new_model.summary()

# Load test images
print('reading test images...')
test_generator = test_datagenerator()

# Get the filenames from the generator
fnames = test_generator.filenames

# Get the ground truth from generator
ground_truth = test_generator.classes

# Get the label to class mapping from the generator
label2index = test_generator.class_indices

# Getting the mapping from class index to class label
idx2label = dict((v, k) for k, v in label2index.items())

# Get the predictions from the model using the generator
print('predicting on the test images...')

prediction_start = time.clock()
predictions = new_model.predict_generator(test_generator,
                                          steps=test_generator.samples / test_generator.batch_size,
                                          verbose=0)

prediction_finish = time.clock()
prediction_time = prediction_finish - prediction_start

predicted_classes = np.argmax(predictions, axis=1)

errors = np.where(predicted_classes != ground_truth)[0]
print("No. of errors = {}/{}".format(len(errors), test_generator.samples))

correct_predictions = np.where(predicted_classes == ground_truth)[0]
print("No. of correct predictions = {}/{}".format(len(correct_predictions), test_generator.samples))

print("Test Accuracy = {0:.2f}%".format(len(correct_predictions)*100/test_generator.samples))

print("Predicted in {0:.3f} minutes!".format(prediction_time/60))

# original = load_img('{}/{}'.format(test_dir, fnames[correct_predictions[2]]))
# print(original)
# # Show the errors
# for i in range(len(errors)):
#     pred_class = np.argmax(predictions[errors[i]])
#     pred_label = idx2label[pred_class]
#
#     title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
#         fnames[errors[i]].split('/')[0],
#         pred_label,
#         predictions[errors[i]][pred_class])
#
#     original = load_img('{}/{}'.format(test_dir, fnames[errors[i]]))
#     plt.figure(figsize=[7, 7])
#     plt.axis('off')
#     plt.title(title)
#     plt.imshow(original)
#     plt.show()
#
# # Show the correct predictions
# for i in range(len(correct_predictions)):
#     pred_class = np.argmax(predictions[correct_predictions[i]])
#     pred_label = idx2label[pred_class]
#
#     title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
#         fnames[correct_predictions[i]].split('/')[0],
#         pred_label,
#         predictions[correct_predictions[i]][pred_class])
#
#     original = load_img('{}/{}'.format(test_dir, fnames[correct_predictions[i]]))
#     plt.figure(figsize=[7, 7])
#     plt.axis('off')
#     plt.title(title)
#     plt.imshow(original)
>>>>>>> 2ca6a6a72573981894106df4e30dd0bd62f05584
#     plt.show()