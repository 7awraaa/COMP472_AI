import json #JSON data
import matplotlib.pyplot as plt #to plot

# Load labels
with open('../data/labeled/labels.json', 'r') as f: #read dataset
    labels = json.load(f)

# Count images per class
class_counts = {} #dictionary for number of images of each class
#iterate over the values (classes) in the loaded labels.
for label in labels.values(): #value=class
    class_counts[label] = class_counts.get(label, 0) + 1#increment the count for each class

#extract class and its value
classes = list(class_counts.keys()) #get the list of classes.
counts = list(class_counts.values()) #get the list of counts corresponding to each class

# Plot bar graph

''' The following lines use the `matplotlib` library in Python to create basic plots. 
    This approach is derived from the tutorial provided by GeeksforGeeks, "Matplotlib tutorial,
    " https://www.geeksforgeeks.org/matplotlib-tutorial/, accessed May 27, 2024.
'''

plt.bar(classes, counts) #create bar graph
plt.xlabel('Class') 
plt.ylabel('Number of Images')
plt.title('Class Distribution')
plt.show()
