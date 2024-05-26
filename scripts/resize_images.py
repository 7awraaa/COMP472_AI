# scripts/resize_images.py

from PIL import Image, ImageEnhance
import os

input_dir = 'data/raw/happy_faces'
output_dir = 'data/cleaned/happy_faces'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = Image.open(os.path.join(input_dir, filename))
        img = img.resize((260, 260))
        
        # Apply slight rotation
        img = img.rotate(5)
        
        # Adjust brightness
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.2)
        
        img.save(os.path.join(output_dir, filename))

print("Images resized and processed successfully.")
