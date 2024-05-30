# COMP472_AI

**Team OB_18**

Alma Almidany (40197854) - Data Specialist

Carmen Derderian (40244084) - Training Specialist

Hawraa Al-Adilee (40246450) - Evaluation Specialist8

**Link to the full dataset**
https://www.kaggle.com/datasets/thienkhonghoc/affectnet

**Purpose of each file**

*class_distribution.py*: This code is used to visualize the images accross different classes in the dataset. It reads a JSON file containing image labels, counts the number of images in each class, and generates a bar graph to display the results. The x-axis represents the classes, and the y-axis represents the number of images in each class.

*label_images.py*: This script sorts images into different folders based on their corresponding emotions and creates a JSON file with the image labels. The images are read from the `data/cleaned/emotion` directory (generated from select_images.py), sorted into their specific emotion folder, and saved in the `data/labeled/emotion` directories. The JSON file, `labels.json`, is used for labeling and to plot the class distribution plotting purposes in other scripts.

*pixel_intensity_distribution.py*: This file is used to compute the pixel distribution of the all photos (.jpg) in each of the four labelled folders by reading the images from the `data/labeled/emotion` directories. The emotion specific images are split into their Red, Green, and Blue channels. After the code completes executing (~15 minutes), a pixel intensity distribution graph is returned with a title corresponding to the folder name: the x-axis represents the pixel intensity and the y-axis represents the the frequency.

*resize_images.py*: This script resizes all images to a consistent dimension of 256x256 pixels for data cleaning. It reads images from the `data/raw/emotion` directories, processes them (including optional brightness adjustments and rotations), and saves the resized images in the `data/cleaned/emotion` directories. It takes in the raw 500 photos of each dataset and applies these modifications ensuring uniformity in the dataset.

*sample_images.py*: This script displays 15 sample images from each class with each image's pixel intensity histogram next to it. It reads images from the `data/labeled/emotion` directories, randomly selects 15 images from each class, and plots them alongside their pixel intensity histograms. 

*select_images*: his script randomly selects 500 images per class from the public AffectNet dataset on Kaggle and stores them in the `data/raw/emotion` directories. It goes through the corresponding subfolders of each emotion and creates a folder labelled with its corresponding folder.

*test_images.py*: This script randomly selects 100 images from each labeled dataset subfolder for testing purposes. It moves the selected images from the `data/labeled` directories to the `data/test` directories, ensuring that they are separate from the training images. The remaining images are moved to the `data/evaluation` directories for further evaluation.

*Notes*: Each emotion has a total of 503 images as there is a picture of each teammate for each emotion.

**To execute the code**

1. ***Ensure the require packages are installed**

    pip install numpy matplotlib pillow scikit-learn

2. **Navigate to the script folder**

    cd scripts

3. **Select Images**

    python select_images.py

4. **Resize Images**

    python resize_images.py

5. **Label Images**

    python label_images.py

6. **Visualize Class Distribution**

    python class_distribution.py

7. **Pixel Intensity Distribution**

    python pixel_intensity_distribution.py

8. **Sample Images**

    python sample_images.py

9. **Select Test Images**

    python test_images.py




