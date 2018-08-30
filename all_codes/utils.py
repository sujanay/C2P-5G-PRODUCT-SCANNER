import json
from keras.preprocessing.image import ImageDataGenerator

def readJSON():
    """
    This function reads the json file created the gui and has the
    neural network training and testing parameters set by the user.
    :return:
    """
    # saved neural network parameters from gui
    jsonFile = 'C:\\Users\\sujanay\\Documents\\NetBeansProjects\\Product_Classifier_GUI\\myjsonfile.json'

    # open the json file and load it's content to "nnparams"
    with open(jsonFile) as f:
        nnparams = json.load(f)

    return nnparams

def showGuiInfo():
    """
    Display user entered parameters
    """
    print("  Training Images Directory:", train_dir)
    print("Validation Images Directory:", validation_dir)
    print("      Test Images Directory:", test_dir)
    print("           Train Batch Size:", train_batchsize)
    print("      Validation Batch Size:", val_batchsize)
    print("            Test Batch Size:", test_batchsize)
    print("                     Epochs:", epochs)
    print("                Show Errors:", show_errors)
    print("   Show Correct Predictions:", show_correct_predictions)


# Read Neural Network Parameters from JSON File
nnparams = readJSON()

image_size = 224                                    # Test Image Size
test_batchsize = int(nnparams['test_batchsize'])    # Test Batch Size
train_batchsize = int(nnparams['train_batchsize'])  # Change the batchsize according to your system RAM
val_batchsize = int(nnparams['valid_batchsize'])    # Validation Batch Size
epochs = int(nnparams['epochs'])
show_errors = nnparams['show_errors']
show_correct_predictions = nnparams['show_correct_predictions']

# Image Dataset Directory
train_dir = nnparams['dataset_directory'] + '\\train'
validation_dir = nnparams['dataset_directory'] + '\\valid'
test_dir = nnparams['dataset_directory'] + '\\test'

if __name__=='__main__':
    showGuiInfo()
