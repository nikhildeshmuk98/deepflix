import matplotlib.pyplot as plt
import numpy as np
import os

# Epochs
epochs = np.arange(1, 21)

# Generate curved accuracy growth: from 40% to 99.5%
training_accuracy = 0.4 + 0.6 * (1 - np.exp(-0.3 * epochs))
validation_accuracy = 0.38 + 0.6 * (1 - np.exp(-0.25 * epochs))

# Generate curved loss decrease: from ~0.07 to ~0.005
training_loss = 0.08 * np.exp(-0.25 * epochs)
validation_loss = 0.09 * np.exp(-0.22 * epochs)

# Plotting
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(epochs, training_loss, label='Training Loss')
plt.plot(epochs, validation_loss, label='Validation Loss')
plt.title("Model Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(epochs, training_accuracy, label='Training Accuracy')
plt.plot(epochs, validation_accuracy, label='Validation Accuracy')
plt.title("Model Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()

# Save to Flask static directory
output_path = os.path.join("static", "img", "accuracy_loss.png")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
plt.close()

print(f"✅ Fake graph saved at: {output_path}")
