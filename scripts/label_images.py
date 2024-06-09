import os #to read write to files, create folders
import json
import shutil #copy/move files

# Input directories
#dictionary mapping each emotion to its corresponding input directory
input_dirs = {
    'happy_faces': '../data/cleaned/happy_faces',
    'neutral_faces': '../data/cleaned/neutral_faces',
    'focused_faces': '../data/cleaned/focused_faces',
    'angry_faces': '../data/cleaned/angry_faces'
}

# Output directories
#dictionary mapping each emotion to its corresponding output directory
output_dirs = {
    'happy_faces': '../data/labeled/happy_faces',
    'neutral_faces': '../data/labeled/neutral_faces',
    'focused_faces': '../data/labeled/focused_faces',
    'angry_faces': '../data/labeled/angry_faces'
}

# Labels
#empty dictionary to store the labels for each image.
labels = {}

#iterate through each face type and its corresponding input directory and map it to its corresponding output directory
for face_type, input_dir in input_dirs.items():
    output_dir = output_dirs[face_type]

    #if output directory does not exist create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #loop through each file in the input directory and adds it to the labels dictionary, associating the filename with its face type.
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"): #to be safe 
            labels[filename] = face_type
            #copy the image file from the input directory to the output directory
            shutil.copy(os.path.join(input_dir, filename), os.path.join(output_dir, filename))
            
''' The line above uses the `copy` function from the `shutil` module to copy files and directories in Python. 
This approach is inspired by the method described in Pynative, "Python Copy Files and Directories,"
https://pynative.com/python-copy-files-and-directories/#:~:text=Suppose%20you%20want%20to%20copy,using%20the%20copy()%20function, accessed May 26, 2024.
'''

# Save labels to a JSON file
''' The following example uses the `json` module in Python to read from and write to JSON files. 
This approach is based on the method described in GeeksforGeeks, "Reading and Writing JSON to a File in Python," 
https://www.geeksforgeeks.org/reading-and-writing-json-to-a-file-in-python/, accessed May 26, 2024.
'''
with open('../data/labeled/labels.json', 'w') as f:
    json.dump(labels, f)
#confirmation
print("Images labeled successfully.")

