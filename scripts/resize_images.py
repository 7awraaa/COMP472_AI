# scripts/resize_images.py

from PIL import Image, ImageEnhance #used for image processing
import os #to read write to files, create folders

#PUT YOU PATH
input_dir = '../data/raw/angry_faces' #set the path for the directory containing the raw images generated from select_images.py
output_dir = '../data/cleaned/angry_faces' #set the path for the directory where the processed images will be saved

#if output directory not created it will create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#loop through each file in input directory, check if it ends with jpg if it does it opens the image file
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"): #to be safe that all are .jpg
        img = Image.open(os.path.join(input_dir, filename))

        ''' The following lines utilizes the `enhance` function from the `ImageEnhance` module in Python to improve image quality. 
        This technique is adapted from the method described in Educative, "How to Enhance Image in Python Pillow," 
        https://www.educative.io/answers/how-to-enhance-image-in-python-pillow, accessed May 26, 2024.
        '''  
    # Adjust brightness
        
       # enhancer = ImageEnhance.Brightness(img)
       # img = enhancer.enhance(1.2)
        
        #Add Rotatation to some images (optional)


        # Resize the image to 256 pixel
        img = img.resize((256, 256))

        #save the resized image in the output directory
        img.save(os.path.join(output_dir, filename))

#confirmation message 
print("Images resized, processed and labeled successfully.")
