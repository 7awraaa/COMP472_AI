# scripts/label_images.py

import os
import json
import shutil

input_dir = 'data/raw/happy_faces'
output_dir = 'data/labeled/happy_faces'
labels = {}

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        labels[filename] = 'happy'
        shutil.move(os.path.join(input_dir, filename), os.path.join(output_dir, filename))

# Save labels to a JSON file
with open(os.path.join(output_dir, 'labels.json'), 'w') as f:
    json.dump(labels, f)

print("Images labeled successfully.")
