# ===========================================================================================================================================
#                                       | Build Image Classifier  |                                      
#                                       | Rock - Paper - Scissors |                                                    
#
# Resources: 
#           https://www.youtube.com/watch?v=u2TjZzNuly8&t=79s
#           https://www.tensorflow.org/datasets/catalog/rock_paper_scissors
#           https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%208%20-%20Lesson%202%20-%20Notebook%20(RockPaperScissors).ipynb#scrollTo=ZABJp7T3VLCU
#                                        
# =============================================================================================================================================

# --------------------------------------- IMPORT LIBRARIES ---------------------------------------------------------
import requests
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
#from keras.models import load_model  
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator     # Image loading and augmentation

# ----------------------------------------------- DOWNLOAD DATASETS ------------------------------------------------

# Download training dataset
url1 = "https://storage.googleapis.com/learning-datasets/rps.zip"
response1 = requests.get(url1, allow_redirects = True)

# Write contents to new file
with open('C:/Users/alexa/OneDrive/Έγγραφα/DifferentialExpressionAnalysis/MachineLearning_tests/tmp/rps.zip', 'wb') as file:
    file.write(response1.content)

# Download test dataset
url2 = "https://storage.googleapis.com/learning-datasets/rps-test-set.zip"
response2 = requests.get(url2, allow_redirects = True)

# Write contents to new file
with open('C:/Users/alexa/OneDrive/Έγγραφα/DifferentialExpressionAnalysis/MachineLearning_tests/tmp/rps-test-set.zip', 'wb') as file:
    file.write(response2.content)


# ----------------------------------------------- UNZIP FILES ---------------------------------------------------------

with zipfile.ZipFile('C:/Users/alexa/OneDrive/Έγγραφα/DifferentialExpressionAnalysis/MachineLearning_tests/tmp/rps.zip', 'r') as zip_ref:
    zip_ref.extractall('C:/Users/alexa/OneDrive/Έγγραφα/DifferentialExpressionAnalysis/MachineLearning_tests/tmp')

with zipfile.ZipFile('C:/Users/alexa/OneDrive/Έγγραφα/DifferentialExpressionAnalysis/MachineLearning_tests/tmp/rps-test-set.zip', 'r') as zip_ref:
    zip_ref.extractall('C:/Users/alexa/OneDrive/Έγγραφα/DifferentialExpressionAnalysis/MachineLearning_tests/tmp')


# --------------------------------------------- CHECK TRAINING DATASET --------------------------------------------------

# Store paths to directories making sure paths are constructed properly with the os.path.join() function
rock_dir = os.path.join('C:/Users/alexa/OneDrive/Έγγραφα/DifferentialExpressionAnalysis/MachineLearning_tests/tmp/rps/rock')
paper_dir = os.path.join('C:/Users/alexa/OneDrive/Έγγραφα/DifferentialExpressionAnalysis/MachineLearning_tests/tmp/rps/paper')
scissors_dir = os.path.join('C:/Users/alexa/OneDrive/Έγγραφα/DifferentialExpressionAnalysis/MachineLearning_tests/tmp/rps/scissors')

# Count images in each directory
print(f"Total number of rock images: {len(os.listdir(rock_dir))}")
print(f"Total number of paper images: {len(os.listdir(paper_dir))}")
print(f"Total number of scissors images: {len(os.listdir(scissors_dir))}")

# List 10 first files in each directory
rock_files = os.listdir(rock_dir)
print(rock_files[:10])

paper_files = os.listdir(paper_dir)
print(paper_files[:10])

scissors_files = os.listdir(scissors_dir)
print(scissors_files[:10])

# Display first 2 images of each directory using matplotlib
pic_index = 2

next_rock = [os.path.join(rock_dir, fname) for fname in rock_files[pic_index-2:pic_index]]
next_paper = [os.path.join(paper_dir, fname) for fname in paper_files[pic_index-2:pic_index]]
next_scissors = [os.path.join(scissors_dir, fname) for fname in scissors_files[pic_index-2:pic_index]]

for i, img_path in enumerate(next_rock + next_paper + next_scissors):
  img = mpimg.imread(img_path)
  plt.imshow(img)
  plt.axis('Off')
  plt.show()

# ------------------------------------ GENERATE IMAGE DATA FROM DIRECTORIES -------------------------------------------------

# Set up training and testing directories
training_dir = 'C:/Users/alexa/OneDrive/Έγγραφα/DifferentialExpressionAnalysis/MachineLearning_tests/tmp/rps'
validation_dir = 'C:/Users/alexa/OneDrive/Έγγραφα/DifferentialExpressionAnalysis/MachineLearning_tests/tmp/rps-test-set'

# Set up training generator: creates training data from images
training_datagen = ImageDataGenerator(rescale = 1./255, #  Scales pixel values to the range [0, 1] from [0, 255], which normalizes the image
                                      rotation_range=40, # Randomly rotates the image up to 40 degrees
                                      width_shift_range=0.2, # Shifts the image horizontally by up to 20% of its width
                                      height_shift_range=0.2, # Shifts the image vertically by up to 20% of its height
                                      shear_range=0.2, # Shears the image by 20%
                                      zoom_range=0.2, # Randomly zooms the image by up to 20%
                                      horizontal_flip=True, # Randomly flips the image horizontally
                                      fill_mode='nearest') #  Fills in any missing pixels after transformations using the nearest pixel values

# Set up validation image data generator - NO AUGMENTATION! Validation pics should be unmodified for evaluation purposes
validation_datagen = ImageDataGenerator(rescale = 1./255) # Normalize image 

# Create training data generator
train_generator = training_datagen.flow_from_directory(training_dir,
                                                       target_size = (150, 150), # Resizes all images to 150 x 150 pixels
                                                       class_mode = 'categorical', # Labels are one-hot encoded
                                                       batch_size = 126) # How many images to process in each batch

# Create validation data generator 
valid_generator = validation_datagen.flow_from_directory(validation_dir,
                                                         target_size = (150,150),
                                                         class_mode = 'categorical',
                                                         batch_size = 126)

# ----------------------------------------------------  MODEL -----------------------------------------------------------------
# Define model
model = tf.keras.models.Sequential([
  # Stack layers in sequence
  # Convolutional Layer 1: 64 filters (kernels) of size (3,3)/ images are 150x150 pixels with 3 channels (RGB color)/ ReLU to introduce non-linearity
  tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (150,150,3)),
  tf.keras.layers.MaxPooling2D(2,2), # For downsampling
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'), # Filters increase (from 64 to 128) to capture more complex patterns
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(), # Rshapes the 3D output from the convolutional layers into a 1D vector to feed into fully connected (dense) layers
  tf.keras.layers.Dropout(0.5), #  Randomly sets 50% of the input units to 0 during training to prevent overfitting
  tf.keras.layers.Dense(512, activation='relu'), # 512 neurons with ReLU activation function
  tf.keras.layers.Dense(3, activation='softmax') # 3 Output neurons for the three classes: ROCK - PAPER - SCISSORS
]) # The softmax activation function ensures that the output is a probability distribution across the 3 classes

# Check model architecture 
model.summary()

# Compile model
model.compile(loss = 'categorical_crossentropy', # Because this is a multi-class classification problem
              optimizer = 'rmsprop', # Adjusts the learning rate based on the magnitude of recent gradients, allowing faster convergence
              metrics = ['accuracy']) # Fraction of correct predictions

# Train model
history = model.fit(train_generator, # Provides the batches of augmented training data from images dataset
                    epochs=25,  # Model will train for 25 complete passes over training dataset
                    steps_per_epoch=20,  # How many batches (steps) are used per epoch 
                    validation_data= valid_generator, # The validation generator provides validation data in batches
                    verbose=1, # Print details for each epoch
                    validation_steps=5) # During validation, 3 batches of validation data will be processed to evaluate model performance after each epoch

# Save model in .h5 format 
model.save("C:/Users/alexa/OneDrive/Έγγραφα/DifferentialExpressionAnalysis/MachineLearning_tests/rps.h5") 

# ------------------------------------------- PLOT MODEL PERFORMANCE METRICS ----------------------------------------------

# Store metrics from model performance using the history.history dictionary
acc = history.history['accuracy'] # Accuracy for the training set over each epoch
val_acc = history.history['val_accuracy'] # Accuracy for the validation set over each epoch
loss = history.history['loss'] # Training loss for each epoch
val_loss = history.history['val_loss'] # Validation loss for each epoch

# Number of epochs
epochs = range(len(acc)) # Used to generate the x-axis values for the plots (0 to len(acc) - 1)
 
'''*PLOTTING TRAINING & VALIDATION ACCURACY* 
The red line represents training accuracy over time.
The blue line represents validation accuracy over time.
By plotting both, we can visually inspect how well the model is performing on the training data vs. the validation data.
If training accuracy increases but validation accuracy plateaus or decreases, it could indicate overfitting,
 where the model is learning to perform well on the training data but not generalizing to unseen data (validation set).
'''

plt.plot(epochs, acc, 'red', label='Training accuracy')
plt.plot(epochs, val_acc, 'blue', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)

# Create figure
plt.figure()

# Display plot 
plt.show()

# ------------------------------------------------ PREDICTION ------------------------------------------------------------
image_dir = 'C:/Users/alexa/OneDrive/Έγγραφα/DifferentialExpressionAnalysis/MachineLearning_tests/images'
image_files = os.listdir(image_dir)

for fn in image_files:

    # Construct the full path to the image
    path = os.path.join(image_dir, fn)
    
    # Load and preprocess image
    img = image.load_img(path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Predict the class of image
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)

    # Output the filename and predicted classes
    print(fn)
    print(classes)



