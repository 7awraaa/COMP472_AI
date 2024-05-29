import json
import matplotlib.pyplot as plt

# Load labels
with open('../data/labeled/labels.json', 'r') as f: #read dataset
    labels = json.load(f)

# Count images per class
class_counts = {} #dictionary for number of images of each class
for label in labels.values(): #value=class
    class_counts[label] = class_counts.get(label, 0) + 1

#extract class and its value
classes = list(class_counts.keys())
counts = list(class_counts.values())

# Plot bar graph
plt.bar(classes, counts) #create bar graph
plt.xlabel('Class') 
plt.ylabel('Number of Images')
plt.title('Class Distribution')
plt.show()
