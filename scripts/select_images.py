# scripts/select_images.py

import os
import random
import shutil

#PUT YOU PATH
input_dir = '/data'
output_dir = '/data'
num_images = 500

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

all_images = [f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')]
selected_images = random.sample(all_images, num_images)

for image in selected_images:
    shutil.copy(os.path.join(input_dir, image), os.path.join(output_dir, image))

print(f"Selected {num_images} images from directory '3' and copied to {output_dir}.")

