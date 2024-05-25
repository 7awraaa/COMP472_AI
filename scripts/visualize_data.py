# scripts/visualize_data.py

import os
import json
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

data_dir = 'data/labeled/happy_faces'
label_file = 'data/labeled/happy_faces/labels.json'

# Load labels
with open(label_file, 'r') as f:
    labels = json.load(f)

# Class Distribution
classes = list(set(labels.values()))
class_counts = {cls: list(labels.values()).count(cls) for cls in classes}

plt.figure(figsize=(10, 5))
plt.bar(class_counts.keys(), class_counts.values())
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Class Distribution')
plt.show()

# Pixel Intensity Distribution per Class
for cls in classes:
    pixels = []
    for filename, label in labels.items():
        if label == cls:
            img = Image.open(os.path.join(data_dir, filename))
            pixels.extend(np.array(img).flatten())
    
    plt.figure()
    plt.hist(pixels, bins=256, color='gray', alpha=0.5)
    plt.title(f'Pixel Intensity Distribution for {cls}')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

# Sample Images with Pixel Intensity Distribution
sample_images = list(labels.keys())[:15]

fig, axs = plt.subplots(5, 3, figsize=(15, 10))
for i, filename in enumerate(sample_images):
    img = Image.open(os.path.join(data_dir, filename))
    axs[i // 3, i % 3].imshow(img)
    axs[i // 3, i % 3].set_title(filename)
plt.show()
