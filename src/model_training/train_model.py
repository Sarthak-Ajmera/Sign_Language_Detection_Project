import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical

# Load processed data
processed_data_path = os.path.join("C:\\", "sign_language_project", "data", "processed")
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

# Verify the shapes
print("y_train shape after one-hot encoding:", y_train.shape)
print("y_test shape after one-hot encoding:", y_test.shape)

# Load the model
model = load_model(os.path.join("C:\\", "sign_language_project", "models", "saved_models", "sign_language_model.h5"))

# Define callbacks
checkpoint = ModelCheckpoint(
    os.path.join("C:\\", "sign_language_project", "models", "saved_models", "best_model.h5"),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stopping]
)

# Save the final model
model.save(os.path.join("C:\\", "sign_language_project", "models", "saved_models", "final_model.h5"))

print("Training complete!")