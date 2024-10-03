# ==================================== Convolutional Neural Networks ============================================
''' *How it works* 
You filter the images before using the deep neural network.
After filtering the images, features within the images could then come to the forefront.
Those features would, then, be used to identify something.

The filters are learned --> they are just parameters.
As an image is fed into the convolutional layer, a number of randomly initialized filters will pass over the image.
=> The results of these are fed into the next layer and maching is performed by the neural network.

Overtime, the filters that give us the image outputs that give the best matches, will be learned. 
THIS PROCESS IS CALLED *"FEATURE EXTRACTION"*
'''

# Import libraries
import tensorflow as tf
from tensorflow import keras

# Define model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', # **
                           input_shape = (28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2,2), # Compress the image & enhance the features
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'), # Stack convolutional layers on top of each other to try to learn from very abstract features (optional)
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(), # We don't specify input shape because we have a convolutional layer on top
    tf.keras.layers.Dense(128, activation = tf.nn.relu),
    tf.keras.layers.Dense(10, activation = tf.nn.softmax)
])


''' ** Specify input shape in convolution layer:
Generate 64 filters and multiply each of them across the image.
Then, each epoch, it will figure out which filters gave the best signals
to help match the images to their labels in the same way it learned 
which parameters worked best in the Dense layer ("ML_test2.py").
'''

