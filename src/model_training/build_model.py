import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Define processed data path
processed_data_path = os.path.join("C:\\", "sign_language_project", "data", "processed")

# Debug: Check if processed data path exists
print("Processed data path:", processed_data_path)
print("Does the path exist?", os.path.exists(processed_data_path))

if os.path.exists(processed_data_path):
    print("Files in processed folder:", os.listdir(processed_data_path))
else:
    raise FileNotFoundError(f"The path {processed_data_path} does not exist.")

# Load processed data
X_train = np.load(os.path.join(processed_data_path, "X_train.npy"))
X_test = np.load(os.path.join(processed_data_path, "X_test.npy"))
y_train = np.load(os.path.join(processed_data_path, "y_train.npy"))
y_test = np.load(os.path.join(processed_data_path, "y_test.npy"))

# Map string labels to integers
label_to_index = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
    'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18,
    'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25,
    'space': 26, 'del': 27, 'nothing': 28  # Add additional labels here
}

# Convert string labels to integers
y_train_int = np.array([label_to_index[label] for label in y_train])
y_test_int = np.array([label_to_index[label] for label in y_test])

# Convert labels to one-hot encoding
num_classes = len(label_to_index)  # Automatically adjusts based on the number of labels
y_train = to_categorical(y_train_int, num_classes)
y_test = to_categorical(y_test_int, num_classes)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),  # First convolutional layer
    MaxPooling2D((2, 2)),  # Pooling layer
    Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer
    MaxPooling2D((2, 2)),  # Pooling layer
    Flatten(),  # Flatten the output for dense layers
    Dense(128, activation='relu'),  # Fully connected layer
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(num_classes, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Save the model architecture
model.save(os.path.join("C:\\", "sign_language_project", "models", "saved_models", "sign_language_model.h5"))

print("Model built and saved!")