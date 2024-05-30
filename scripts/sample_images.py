import os #to read write to files, create folders
import numpy as np #numerical operations.
import matplotlib.pyplot as plt #to plot
from PIL import Image #to work with images


''' The blog post by se7entyse7en.dev was used to better understand the theory behind image histograms. 
The post, "Understanding Image Histograms with OpenCV," is available at 
https://se7entyse7en.dev/posts/understanding-image-histograms-with-opencv/ accessed May 28, 2024.
'''

# Define directories
data_dir = '../data/labeled'
classes = ['happy_faces', 'angry_faces', 'neutral_faces', 'focused_faces']  #list of classes can do one by one also  


# Initialize a dictionary to hold pixel intensity values for each class and each channel
# there's keys for each class, each class has 3 channels, they are initially empty lists
pixel_intensities = {cls: {channel: [] for channel in ['Red', 'Green', 'Blue']} for cls in classes}

# Function to plot image and histogram
''' The approach for plotting graphs of average pixel intensities along a line of an image is inspired by the discussion on 
Stack Overflow, "python - how to get average pixel intensities along a line of an image and plot them on a graph?," available at 
https://stackoverflow.com/questions/59297798/python-how-to-get-average-pixel-intensities-along-a-line-of-an-image-and-plot (accessed May 27, 2024).

Furthermore, the code to plot a pixel intensity distribution graph are based on the tutorial provided by GeeksforGeeks,
"OpenCV python program to analyze an image using histogram," available at 
https://www.geeksforgeeks.org/opencv-python-program-analyze-image-using-histogram/ accessed May 27, 2024.
and the tutorial from Analytics Vidhya, "Advanced OpenCV: BGR Pixel Intensity Plots," 
available at https://www.analyticsvidhya.com/blog/2022/05/advanced-opencv-bgr-pixel-intensity-plots/ accessed May 27,2024.
'''


#function that takes an image (img), pixel intensities for each channel (intensities), and a title (title) as input. 
#plot the image and histograms side by side

def plot_image_and_histogram(img, intensities, title):
    #create figure, an array of 2 subplots with 1 row and 2 colums and set figure size to 10X4
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    #plot image on first subplot
    axs[0].imshow(img)
    axs[0].axis('off') #turn off labels because image

    #loop through each channel and its corresponding intensities from intesities dictionary
    for channel, intensities in intensities.items():

    #Plots a histogram of pixel intensities for the current channel on the second subplot. 
        axs[1].hist(intensities, bins=256, range=(0, 255), alpha=0.5, label=channel)
    axs[1].set_title(title)
    axs[1].set_xlabel('Pixel Intensity')
    axs[1].set_ylabel('Frequency')
    axs[1].legend()
    plt.show()


#outside function 
# Loop through each class
for cls in classes:
    class_dir = os.path.join(data_dir, cls) #get directory
    # Select 15 random images
    image_files = np.random.choice(os.listdir(class_dir), 15, replace=False)
    
    #for eahc image, open it ,convert to NumyPy array and flatten chanel into 1d array
    for filename in image_files:
        if filename.endswith('.jpg'):
            img_path = os.path.join(class_dir, filename)
            img = Image.open(img_path)
            pixels = np.array(img)

            ''' The method for flattening a 2D NumPy array into a 1D array is based on the tutorial from 
                GeeksforGeeks, "Python | Flatten a 2d numpy array into 1d array," available at 
                https://www.geeksforgeeks.org/python-flatten-a-2d-numpy-array-into-1d-array/ accessed May 27, 2024.
            '''

            # Split the image into its Red, Green, and Blue channels
            red_channel = pixels[:, :, 0].flatten() #extract the red channel of the image by selecting all rows and columns (:) and the first (index 0) channel. The flatten() method converts the 2D array into a 1D array
            green_channel = pixels[:, :, 1].flatten()
            blue_channel = pixels[:, :, 2].flatten()

            # Extend the lists of pixel intensities for each channel and each class

            ''' The method for splitting RGB channels in Python is based on the tutorial from 
                Educative, "Splitting RGB Channels in Python," available at 
                https://www.educative.io/answers/splitting-rgb-channels-in-python (accessed date).
            '''
            pixel_intensities[cls]['Red'].extend(red_channel) # extend by appending the pixel values from the flattened red channel array
            pixel_intensities[cls]['Green'].extend(green_channel)
            pixel_intensities[cls]['Blue'].extend(blue_channel)

            # Plot image and histogram by calling function
            plot_image_and_histogram(img, pixel_intensities[cls], f'{cls.replace("_", " ").title()} Sample Images and Histograms')
