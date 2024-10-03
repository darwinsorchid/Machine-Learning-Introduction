# ==================================== Basic Machine Learning Practical Example ============================================

# Import libraries
import keras
import numpy as np

# Define model
model = keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])])

# Compile model
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')

# Data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype = float)

# Answers
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype = float)

# Train model: go through loop 500 times
model.fit(xs, ys, epochs = 500)

# Use model to make a prediction
print(model.predict(np.array([10.0])))