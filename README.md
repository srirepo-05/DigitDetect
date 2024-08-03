# MNIST Digit Classification

This project demonstrates a basic implementation of a neural network using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. The model achieves high accuracy in predicting the digits and includes steps for training, testing, and evaluating the performance.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Visualization](#visualization)

## Introduction

The MNIST dataset contains 60,000 training images and 10,000 testing images of handwritten digits (0-9). Each image is 28x28 pixels. This project uses a simple neural network to classify the digits with high accuracy.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/mnist-digit-classification.git
    cd mnist-digit-classification
    ```
    

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
    

## Usage

1. Import the necessary libraries:

    ```python
    import os
    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    ```

2. Load and normalize the MNIST dataset:

   ```python
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)
   ```
    

## Model Training

1. Define and compile the model:

    ```python
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    ```
    

2. Train the model:

    ```python
    model.fit(x_train, y_train, epochs=10)
    model.save('digits.model')
    ```
    

## Evaluation

Evaluate the model on the test dataset:

```python
loss, accuracy = model.evaluate(x_test, y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)
```


## Testing

Test the model with custom images:

```python
for i in range(1, 20):
    file_name = f"/content/digits/digit{i}.png"
    img = cv2.imread(file_name)[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f"The number is probably a {np.argmax(prediction)}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
    
  ```


## Visualization

Visualize the model's accuracy and loss over epochs:

```python
history = model.fit(x_train, y_train, epochs=10)
plt.plot(history.history['accuracy'])
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(['train'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(['train'], loc='upper left')
plt.show()
```
