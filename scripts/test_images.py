# scripts/test_images.py

# Select 100 random test images

import os
import random
import shutil

# PUT YOUR PATH
input_dir = '/data/labeled'
output_dir = '/data/test'
evaluation_dir = '/data/evaluation'
num_images = 100

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

all_images = [f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')]
selected_images = random.sample(all_images, num_images)

for image in selected_images:
    shutil.move(os.path.join(input_dir, image), os.path.join(output_dir, image))

# Move rest of images to the evaluation directory
remaining_images = [f for f in os.listdir(input_dir) if f not in selected_images]
for image in remaining_images:
    shutil.move(os.path.join(input_dir, image), os.path.join(evaluation_dir, image))

print(f"Selected {num_images} images from {input_dir} and moved to {output_dir}.")
