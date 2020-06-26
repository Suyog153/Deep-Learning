import tensorflow as tf
import numpy as np
# Selecting mnist handwritten numbers' dataset
data = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = data.load_data()   # Loading variables in dataset
x_train, x_test = x_train/255.0, x_test/255.0             # normalization of training and test data

## Creating a Sequential model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape= (28, 28)),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(10, activation = 'softmax')])

model.compile(optimizer='adam',
              loss= 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training a model
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

# Making predictions on trained model.
predictions = model.predict(x_test)
print(np.argmax(predictions[0]))




