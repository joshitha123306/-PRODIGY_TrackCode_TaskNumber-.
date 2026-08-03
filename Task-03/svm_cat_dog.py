import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ==========================
# Dataset Path
# ==========================

dataset_path = r"archive\training_set\training_set"

categories = ["cats", "dogs"]

print("Dataset exists :", os.path.exists(dataset_path))
print("Cats folder    :", os.path.exists(os.path.join(dataset_path, "cats")))
print("Dogs folder    :", os.path.exists(os.path.join(dataset_path, "dogs")))

# ==========================
# Load Images
# ==========================

data = []
labels = []

IMG_SIZE = 64

print("\nLoading images...\n")

for label, category in enumerate(categories):

    folder = os.path.join(dataset_path, category)

    print(f"Reading images from: {folder}")

    for image_name in os.listdir(folder):

        image_path = os.path.join(folder, image_name)

        try:
            image = cv2.imread(image_path)

            if image is None:
                continue

            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            data.append(image.flatten())
            labels.append(label)

        except Exception as e:
            print("Skipped:", image_path)

print("\nImages Loaded Successfully!")

# ==========================
# Convert to NumPy Arrays
# ==========================

X = np.array(data)
y = np.array(labels)

print("Total Images :", len(X))

# ==========================
# Split Dataset
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Train SVM
# ==========================

print("\nTraining SVM...\n")

model = SVC(kernel="linear")

model.fit(X_train, y_train)

print("Training Completed!")

# ==========================
# Prediction
# ==========================

predictions = model.predict(X_test)

# ==========================
# Accuracy
# ==========================

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy : {:.2f}%".format(accuracy * 100))

print("\nClassification Report\n")

print(classification_report(
    y_test,
    predictions,
    target_names=categories
))