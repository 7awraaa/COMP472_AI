# scripts/test_images.py

# Select 100 random test images

import os #to read write to files, create folders
import random #generate random numbers
import shutil #copy/move files

# PUT YOUR PATH
input_dir = '/data/labeled' #set the path for the directory containing the labeled images
output_dir = '/data/test' #set the path for the directory where the selected test images will be moved
evaluation_dir = '/data/evaluation' #set the path for the directory where the remaining images will be moved

num_images = 100 #number of images randomly selected 

''' The following lines uses the `move` function from the `shutil` module to move files and directories in Python. 
This approach is inspired by the method described in Pynative, "Python Copy Files and Directories,"
https://pynative.com/python-copy-files-and-directories/#:~:text=Suppose%20you%20want%20to%20copy,using%20the%20copy()%20function, accessed May 26, 2024.
'''

#Checks if the output directory exists if not it creates the directory using makedirs
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
 


''' The following line uses list comprehension to filter and list all image files with extensions .jpg in the input directory. 
    This approach is inspired by the method described in GeeksforGeeks, "How to use os with python list comprehension,
    https://www.geeksforgeeks.org/how-to-use-os-with-python-list-comprehension/, accessed May 26, 2024 
'''

#create list called all_images that has all the filenames of the images in the input directory that end with .jpg using list comprehension
all_images = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
selected_images = random.sample(all_images, num_images) #select 100 random images from list above

#loop through all images in list and move each selected image from input to output directory
for image in selected_images:
    shutil.move(os.path.join(input_dir, image), os.path.join(output_dir, image))


''' The following line has been prompted by ChatGpt.
    OpenAI, "ChatGPT: Chat Generative Pre-trained Transformer," OpenAI, San Francisco, CA, 2024. [Online]. Available: https://chat.openai.com/. [Accessed: May 28, 2024].
    Prompt: “How to store the remaining pictures in data/evaluation?” 
    Response: 
    “remaining_images = [f for f in os.listdir(input_dir) if f not in selected_images] for image in remaining_images: shutil.move(os.path.join(input_dir, image), os.path.join(evaluation_dir, image))”
'''

# Move rest of images to the evaluation directory
#using list comprehension create list of the remaining images by going all images from input directory and cheking if they are part of selected_images list. 
#part of list only if the picture is not already in selected_images.
remaining_images = [f for f in os.listdir(input_dir) if f not in selected_images]

#loop through all images in list and move remaining_images to evaluation dircetory
for image in remaining_images:
    shutil.move(os.path.join(input_dir, image), os.path.join(evaluation_dir, image))

#confirmation message
print(f"Selected {num_images} images from {input_dir} and moved to {output_dir}.")
