import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define directories
data_dir = '../../../data/labeled/'
classes = ['happy_faces']  #also neutral_faces, angry_faces, focused_faces

# Initialize a dictionary to hold pixel intensity values for each class and each channel
pixel_intensities = {cls: {channel: [] for channel in ['Red', 'Green', 'Blue']} for cls in classes}

# Loop through each class
for cls in classes:
    class_dir = os.path.join(data_dir, cls)
    for filename in os.listdir(class_dir):
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

# Plot histograms
for cls, channels in pixel_intensities.items():
    plt.figure(figsize=(10, 6))
    # Define the colors for each channel
    colors = {'Red': 'red', 'Green': 'green', 'Blue': 'blue'}
    for channel, intensities in channels.items():
        plt.hist(intensities, bins=256, range=(0, 255), alpha=0.5, label=channel, color=colors[channel])
    plt.title(f'Pixel Intensity Distribution for {cls.replace("_", " ").title()}')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()