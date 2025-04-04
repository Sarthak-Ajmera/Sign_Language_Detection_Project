import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Define paths
raw_data_path = os.path.join("C:\\", "sign_language_project", "data", "raw", "asl_alphabet_train")
processed_data_path = os.path.join("C:\\", "sign_language_project", "data", "processed")

# Debug: Check if raw data path exists
print("Raw data path:", raw_data_path)
print("Does the path exist?", os.path.exists(raw_data_path))

# Create processed data folders
os.makedirs(os.path.join(processed_data_path, "train"), exist_ok=True)
os.makedirs(os.path.join(processed_data_path, "test"), exist_ok=True)

# Parameters
image_size = (64, 64)  # Resize images to 64x64

def preprocess_images(folder_path, output_folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            # Read image
            img = cv2.imread(os.path.join(folder_path, filename))
            # Resize image
            img = cv2.resize(img, image_size)
            # Normalize pixel values to [0, 1]
            img = img / 255.0
            # Save processed image
            output_path = os.path.join(output_folder, label, filename)
            os.makedirs(os.path.join(output_folder, label), exist_ok=True)
            cv2.imwrite(output_path, img * 255)  # Save as 8-bit image
            images.append(img)
            labels.append(label)
    return images, labels

# Preprocess all classes
all_images = []
all_labels = []
for class_name in os.listdir(raw_data_path):
    class_path = os.path.join(raw_data_path, class_name)
    if os.path.isdir(class_path):
        print(f"Processing {class_name}...")
        images, labels = preprocess_images(class_path, os.path.join(processed_data_path, "train"), class_name)
        all_images.extend(images)
        all_labels.extend(labels)

# Convert to numpy arrays
all_images = np.array(all_images)
all_labels = np.array(all_labels)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

# Save processed data
np.save(os.path.join(processed_data_path, "X_train.npy"), X_train)
np.save(os.path.join(processed_data_path, "X_test.npy"), X_test)
np.save(os.path.join(processed_data_path, "y_train.npy"), y_train)
np.save(os.path.join(processed_data_path, "y_test.npy"), y_test)

print("Data preprocessing complete!")