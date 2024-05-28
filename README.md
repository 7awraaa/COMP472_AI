# COMP472_AI
Team OB_18

Alma Almidany (40197854) - Data Specialist

Carmen Derderian (40244084) - Training Specialist

Hawraa Al-Adilee (40246450) - Evaluation Specialist

Purpose of each file

*select_images*: The program in that file is used to extract 500 photos randomy from the public dataset available on Kaggle website. It goes through the corresponding subfolders of each emotion and creates a folder labelled with its corresponding folder. It stores the images in a folder respecting the following directory: data/raw/emotion.

*resize_images.py*: The code of this folder assists in the data cleaning as it adjusts brightness and resizes all images to match the chosen picture size (256x256 pixels). It takes in the raw 500 photos of each dataset and applies these modifications. It takes folder from data/raw/emotion and stores the modified images in a folder respecting the following directory: data/cleaned/emotion. 

*label_images.py*: This script takes the folders stored in cleaned/ and puts each emotion in its own folder. The output respects the following directory: data/labeled/emotion. The process creates a json file simultaneously, which will be later used to plot the required graphs. 

*pixel_intensity_distribution.py*: This file is used to compute the pixel distribution of the all photos (.jpg) in each of the four labelled folders. After the code completes executing (~15 minutes), a pixel intensity distribution graph is returned with a title corresponding to the folder name: the horizontal axis represents the pixel intensity and the vertical axis represents the the frequency. 

*class_dsitribution.py*: This code is used to visualize the images accross different classes in the dataset. The code generates a bar graph. 

*test_images.py*: This script randomly selects 100 pictures out of the 500 ones from each of the labelled dataset subfolders for testing.