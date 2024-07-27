import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Loading the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalization of pixel values
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0


# converting the labels into one-hot encoded verctors
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

#CIFAR-10 dataset class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Creating a  data augmentation generator
datagenerator = ImageDataGenerator(rotation_range=15,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True)

# Fiting the generator on the training data
X_gen = datagenerator.fit(X_train)

# Initializing the CNN model
model = Sequential()

# Adding convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flattening the output before moving it into the fully connected layers
model.add(Flatten())

# Adding fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# Compile the CNN model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define the number of epochs and the batch size
epochs = 50
batch_size = 64

# Train the model
history = model.fit(datagenerator.flow(X_train, y_train, batch_size=batch_size),steps_per_epoch=len(X_train) // batch_size,epochs=epochs,validation_data=(X_test, y_test))

# Evaluating the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)

# Plotting the training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotting the training and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Choossing 10 random test images for visualization
num_samples = 10
random_indices = np.random.randint(0, len(X_test), num_samples)
random_images = X_test[random_indices]

# Make predictions
predictions = model.predict(random_images)

# Plot the random test images along with their predicted classes
for i in range(num_samples):
    plt.imshow(random_images[i])
    plt.title(f"Predicted: {class_names[np.argmax(predictions[i])]}, True: {class_names[np.argmax(y_test[random_indices[i]])]}")
    plt.axis('off')
    plt.show()

# Save the trained model
model.save('cifar10_model.h5')

