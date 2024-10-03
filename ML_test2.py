# ==================================== Basic Computer Vision using a Deep Neural Network ============================================

# Import libraries
import tensorflow as tf
from tensorflow import keras

# Load dataset
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels) , (test_images, test_labels) = fashion_mnist.load_data()

# Define model
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)), # First Layer: size of images (28 x 28 pixels)
    keras.layers.Dense(128, activation = tf.nn.relu), # Second Layer: 128 functions with parameters that the model will have to predict
    keras.layers.Dense(10, activation = tf.nn.softmax) # Last Layer: number of clothing types
])


# Compile model
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])


# Training the model: fit train images to train labels for 5 loops
model.fit(train_images, train_labels, epochs = 5)

# Test how well model performs
test_loss , test_acc = model.evaluate(test_images, test_labels)
print(f"Test loss = {test_loss}. Tets accuracy = {test_acc}")

# Use model to predict image
