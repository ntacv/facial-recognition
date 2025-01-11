import matplotlib.pyplot as plt
import os
import csv

# Load and parse the history file
history = {
    'accuracy': [],
    'loss': [],
    'val_accuracy': [],
    'val_loss': []
}

with open('output_dir/cnn_face_emotion.model.history.csv', 'r') as f:
    csv_reader = csv.DictReader(f)
    for row in csv_reader:
        history['accuracy'].append(float(row['accuracy']))
        history['loss'].append(float(row['loss']))
        history['val_accuracy'].append(float(row['val_accuracy']))
        history['val_loss'].append(float(row['val_loss']))
# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot accuracy
epochs = range(1, len(history['accuracy']) + 1)
ax1.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy')
if 'val_accuracy' in history:
    ax1.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.grid(True)
ax1.legend()

# Plot loss
ax2.plot(epochs, history['loss'], 'b-', label='Training Loss')
if 'val_loss' in history:
    ax2.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.grid(True)
ax2.legend()

# Adjust layout and save
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()
os.system('xdg-open training_history.png')
