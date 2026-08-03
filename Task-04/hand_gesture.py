import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# -----------------------------
# DATASET PATH
# -----------------------------
dataset_path = "leapGestRecog"

images = []
labels = []
gesture_names = {}

label = 0

# -----------------------------
# READ DATASET
# -----------------------------
for person in os.listdir(dataset_path):

    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    for gesture in os.listdir(person_path):

        gesture_path = os.path.join(person_path, gesture)

        if not os.path.isdir(gesture_path):
            continue

        gesture_names[label] = gesture

        for file in os.listdir(gesture_path):

            image_path = os.path.join(gesture_path, file)

            img = cv2.imread(image_path)

            if img is None:
                continue

            img = cv2.resize(img, (64, 64))

            images.append(img)

            labels.append(label)

        label += 1

# -----------------------------
# CHECK DATASET
# -----------------------------
print("Images Loaded :", len(images))
print("Labels Loaded :", len(labels))

# -----------------------------
# CONVERT TO NUMPY
# -----------------------------
images = np.array(images, dtype="float32")
labels = np.array(labels)

images = images / 255.0

labels = to_categorical(labels)

# -----------------------------
# SPLIT DATA
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("Training Images :", X_train.shape)
print("Testing Images :", X_test.shape)

# -----------------------------
# CNN MODEL
# -----------------------------
model = Sequential()

model.add(Conv2D(32, (3,3), activation="relu", input_shape=(64,64,3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128, activation="relu"))

model.add(Dense(labels.shape[1], activation="softmax"))

# -----------------------------
# COMPILE MODEL
# -----------------------------
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining Started...\n")

# -----------------------------
# TRAIN MODEL
# -----------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

print("\nTraining Completed Successfully\n")

# -----------------------------
# EVALUATE MODEL
# -----------------------------
loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy :", accuracy)

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("model.keras")

print("Model Saved Successfully!")

# -----------------------------
# ACCURACY GRAPH
# -----------------------------
plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Accuracy Graph")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.show()

# -----------------------------
# LOSS GRAPH
# -----------------------------
plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Loss Graph")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.show()