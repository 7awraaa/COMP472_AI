import os
import json
import shutil

# Input directories
input_dirs = {
    'happy_faces': '../data/raw/happy_faces',
    'neutral_faces': '../data/raw/neutral_faces',
    'focused_faces': '../data/raw/focused_faces',
    'angry_faces': '../data/raw/angry_faces'
}

# Output directories
output_dirs = {
    'happy_faces': '../data/labeled/happy_faces',
    'neutral_faces': '../data/labeled/neutral_faces',
    'focused_faces': '../data/labeled/focused_faces',
    'angry_faces': '../data/raw/angry_faces'
}

# Labels
labels = {}

for face_type, input_dir in input_dirs.items():
    output_dir = output_dirs[face_type]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            labels[filename] = face_type
            shutil.copy(os.path.join(input_dir, filename), os.path.join(output_dir, filename))

# Save labels to a JSON file
with open('../data/labeled/labels.json', 'w') as f:
    json.dump(labels, f)

print("Images labeled successfully.")

