import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define directories
data_dir = '../data/labeled'
classes = ['happy_faces', 'angry_faces', 'neutral_faces', 'focused_faces']  # Add other classes 

# Initialize a dictionary to hold pixel intensity values for each class and each channel
pixel_intensities = {cls: {channel: [] for channel in ['Red', 'Green', 'Blue']} for cls in classes}

# Function to plot image and histogram
def plot_image_and_histogram(img, intensities, title):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(img)
    axs[0].axis('off')
    for channel, intensities in intensities.items():
        axs[1].hist(intensities, bins=256, range=(0, 255), alpha=0.5, label=channel)
    axs[1].set_title(title)
    axs[1].set_xlabel('Pixel Intensity')
    axs[1].set_ylabel('Frequency')
    axs[1].legend()
    plt.show()

# Loop through each class
for cls in classes:
    class_dir = os.path.join(data_dir, cls)
    # Select 15 random images
    image_files = np.random.choice(os.listdir(class_dir), 15, replace=False)
    for filename in image_files:
        if filename.endswith('.jpg'):
            img_path = os.path.join(class_dir, filename)
            img = Image.open(img_path)
            pixels = np.array(img)

            # Split the image into its Red, Green, and Blue channels
            red_channel = pixels[:, :, 0].flatten()
            green_channel = pixels[:, :, 1].flatten()
            blue_channel = pixels[:, :, 2].flatten()

            # Extend the lists of pixel intensities for each channel and each class
            pixel_intensities[cls]['Red'].extend(red_channel)
            pixel_intensities[cls]['Green'].extend(green_channel)
            pixel_intensities[cls]['Blue'].extend(blue_channel)

            # Plot image and histogram
            plot_image_and_histogram(img, pixel_intensities[cls], f'{cls.replace("_", " ").title()} Sample Images and Histograms')
