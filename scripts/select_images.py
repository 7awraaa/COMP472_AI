# scripts/select_images.py
#select 500 images randomly from the raw database


import os #to read write to files, create folders
import random #generate random numbers
import shutil #copy/move files

#PUT YOU PATH
input_dir = '../AffectNet/train/3' #set the path for the directory containing the raw images from the complete dataset
output_dir = '../data/raw/happy_faces' #set the path for the directory where the selected images will be copied which will become our dataset

num_images = 500 #number of images randomly selected

#if output directory not created it will create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

''' The following line uses list comprehension to filter and list all image files with extensions .jpg in the input directory. 
    This approach is inspired by the method described in GeeksforGeeks, "How to use os with python list comprehension,
    https://www.geeksforgeeks.org/how-to-use-os-with-python-list-comprehension/, accessed May 26, 2024 
'''
#create list called all_images that has all the filenames of the images in the input directory that end with .jpg using list comprehension
all_images = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
selected_images = random.sample(all_images, num_images) #select 500 random images from list above

''' The following line uses the `copy` function from the `shutil` module to copy files and directories in Python. 
This approach is inspired by the method described in Pynative, "Python Copy Files and Directories,"
https://pynative.com/python-copy-files-and-directories/#:~:text=Suppose%20you%20want%20to%20copy,using%20the%20copy()%20function, accessed May 26, 2024.
'''
#loop through all images in list and move each selected image from input to output directory
for image in selected_images:
    shutil.copy(os.path.join(input_dir, image), os.path.join(output_dir, image))

#confirmation message
print(f"Selected {num_images} images from {input_dir} and copied to {output_dir}.")

