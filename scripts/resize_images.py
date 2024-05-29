# scripts/resize_images.py

from PIL import Image, ImageEnhance
import os

#PUT YOU PATH
input_dir = '../data/raw/angry_faces'
output_dir = '../data/cleaned/angry_faces'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        img = Image.open(os.path.join(input_dir, filename))


        # Adjust brightness
       # enhancer = ImageEnhance.Brightness(img)
       # img = enhancer.enhance(1.2)
        #Add Rotatation to some images

        # Resize the image
        img = img.resize((256, 256))

        img.save(os.path.join(output_dir, filename))

print("Images resized, processed and labeled successfully.")
