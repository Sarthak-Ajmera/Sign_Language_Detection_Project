import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load processed data
processed_data_path = os.path.join("C:\\", "sign_language_project", "data", "processed")
X_test = np.load(os.path.join(processed_data_path, "X_test.npy"))
y_test = np.load(os.path.join(processed_data_path, "y_test.npy"))

# Load the best model
model = load_model(os.path.join("C:\\", "sign_language_project", "models", "saved_models", "best_model.h5"))

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Map string labels to integers
label_to_index = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
    'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18,
    'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25,
    'space': 26, 'del': 27, 'nothing': 28  # Add additional labels here
}

# Convert string labels to integers
y_test_classes = np.array([label_to_index[label] for label in y_test])

# Classification report
print("Classification Report:")
print(classification_report(y_test_classes, y_pred_classes))

# Confusion matrix
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig(os.path.join("C:\\", "sign_language_project", "results", "confusion_matrix.png"))
plt.show()

print("Evaluation complete! Check the confusion matrix in the 'results' folder.")