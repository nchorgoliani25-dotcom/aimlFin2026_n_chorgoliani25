# Convolutional Neural Networks (CNNs)

## Description
Convolutional Neural Networks (CNNs) are a type of deep learning neural network that automatically extract hierarchical features from visual or structured data. CNNs are widely used in cybersecurity for phishing detection, malware recognition, and anomaly detection. This practical example demonstrates a CNN using a small synthetic dataset.

## Practical Example: Synthetic Phishing Image Classification

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Generate synthetic data
num_samples = 100
img_size = 28
X = np.random.rand(num_samples, img_size, img_size, 1)
y = np.random.randint(0, 2, size=(num_samples,))
y_cat = to_categorical(y, 2)

# Build CNN
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(img_size,img_size,1)),
    MaxPooling2D((2,2)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X, y_cat, epochs=5, batch_size=10)
