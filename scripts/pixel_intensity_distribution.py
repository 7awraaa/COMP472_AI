import os #to read write to files, create folders
import numpy as np #numerical operations.
import matplotlib.pyplot as plt #to plot
from PIL import Image #to work with images

''' ChatGpt was used to understand the concepts of analyzing an image using a histogram.
    OpenAI, "ChatGPT: Chat Generative Pre-trained Transformer," OpenAI, San Francisco, CA, 2024. [Online]. Available: https://chat.openai.com/. [Accessed: May 27, 2024]. 
    OpenAI provided assistance in understanding the concepts for accurate application in the project. 

    The blog post by se7entyse7en.dev was used to better understand the theory behind image histograms. 
    The post, "Understanding Image Histograms with OpenCV," is available at 
    https://se7entyse7en.dev/posts/understanding-image-histograms-with-opencv/ accessed May 28, 2024.

    We used the YouTube video to gain further insights into the topic. 
    The video is available at "Understanding Image Histograms with OpenCV," YouTube, 
    https://youtu.be/kSqxn6zGE0c?si=3rEk9othGLyyzKV3 accessed May 28, 2024.
    
'''

# Define directories
data_dir = '../../../data/labeled/' #directory to the labeled data depends on current directory path when running the script
classes = ['happy_faces']  #also neutral_faces, angry_faces, focused_faces

# Initialize a dictionary to hold pixel intensity values for each class and each channel
# there's keys for each class, each class has 3 channels, they are initially empty lists
pixel_intensities = {cls: {channel: [] for channel in ['Red', 'Green', 'Blue']} for cls in classes}

# Loop through each class
for cls in classes:
    class_dir = os.path.join(data_dir, cls) #get the directory path for the current class
   
    #loop through each image file in the class directory
    for filename in os.listdir(class_dir):
        if filename.endswith('.jpg'): #to be safe

            #open image and convert to NumPy array for processing
            img_path = os.path.join(class_dir, filename) #full path to current image file
            img = Image.open(img_path)
            pixels = np.array(img)

            # Split the image into its Red, Green, and Blue channels

            ''' The method for flattening a 2D NumPy array into a 1D array is based on the tutorial from 
                GeeksforGeeks, "Python | Flatten a 2d numpy array into 1d array," available at 
                https://www.geeksforgeeks.org/python-flatten-a-2d-numpy-array-into-1d-array/ accessed May 27, 2024.
            '''
            red_channel = pixels[:, :, 0].flatten()  #extract the red channel of the image by selecting all rows and columns (:) and the first (index 0) channel. The flatten() method converts the 2D array into a 1D array
            green_channel = pixels[:, :, 1].flatten() #flatten green values
            blue_channel = pixels[:, :, 2].flatten() # flatten blue values

            # Extend the lists of pixel intensities for each channel and each class

            ''' The method for splitting RGB channels in Python is based on the tutorial from 
                Educative, "Splitting RGB Channels in Python," available at 
                https://www.educative.io/answers/splitting-rgb-channels-in-python (accessed date).
            '''
            pixel_intensities[cls]['Red'].extend(red_channel) # extend by appending the pixel values from the flattened red channel array
            pixel_intensities[cls]['Green'].extend(green_channel)
            pixel_intensities[cls]['Blue'].extend(blue_channel)

# Plot histograms
            
    ''' The code for plotting graphs is based on the tutorial provided by GeeksforGeeks,
        "OpenCV python program to analyze an image using histogram," available at 
        https://www.geeksforgeeks.org/opencv-python-program-analyze-image-using-histogram/ accessed May 27, 2024.
        and the tutorial from Analytics Vidhya, "Advanced OpenCV: BGR Pixel Intensity Plots," 
        available at https://www.analyticsvidhya.com/blog/2022/05/advanced-opencv-bgr-pixel-intensity-plots/ accessed May 27,2024.
    '''

    for cls, channels in pixel_intensities.items():
        plt.figure(figsize=(10, 6)) #set the figure size

    # Define the colors for each channel
    colors = {'Red': 'red', 'Green': 'green', 'Blue': 'blue'}

    #loop through each channel in the class
    for channel, intensities in channels.items():
        plt.hist(intensities, bins=256, range=(0, 255), alpha=0.5, label=channel, color=colors[channel])
    plt.title(f'Pixel Intensity Distribution for {cls.replace("_", " ").title()}')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show() #display