import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import torch.nn as nn

# CNN models
# All three models must be defined

class OptimizedCNN(nn.Module):
    def __init__(self):
        super(OptimizedCNN, self).__init__()
        # Define the convolutiona layers and batch normalization layers as defined in the Main Model
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),  # New convolutional layer
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Get the output size of the convolutional layers
        self.fc_input_size = self._get_conv_output_size(torch.randn(1, 3, 256, 256))  # Update input size
        # Define the fully connected layers
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.fc_input_size, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 4)
        )

    def _get_conv_output_size(self, x):
        # Pass the input through the convolutional layers to determine the output size
        x = self.conv_layer(x)
        return x.view(x.size(0), -1).size(1)

    def forward(self, x):
        # Define the forward pass through the network
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

class Variant1CNN(nn.Module):
    def __init__(self):
        super(Variant1CNN, self).__init__()
        # Define the convolutiona layers and batch normalization layers as defined in Variant 1
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # New convolutional layer which distinguishes Variant 1
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            
        )
        # Get the output size of the convolutional layers to know the depth
        self.fc_input_size = self._get_conv_output_size(torch.randn(1, 3, 256, 256))
        # Define the fully connected layers
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.fc_input_size, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 4)
        )

    def _get_conv_output_size(self, x):
        # Pass the input through the convolutional layers to determine the output size
        x = self.conv_layer(x)
        return x.view(x.size(0), -1).size(1)

    def forward(self, x):
        # Define the forward pass through the network
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x
    
class Variant2CNN(nn.Module):
    def __init__(self):
        super(Variant2CNN, self).__init__()
        # Define the convolutiona layers and batch normalization layers as defined in Variant 2
        # Change kernel size to 5 in each layer
        # Change padding to 2 in each layer
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Number of convolutional layers is the same as in the Main Model
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, padding=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            
        )
        # Get the output size of the convolutional layers to know the depth
        self.fc_input_size = self._get_conv_output_size(torch.randn(1, 3, 256, 256))
        # Define the fully connected layers
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.fc_input_size, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 4)
        )

    def _get_conv_output_size(self, x):
        # Pass the input through the convolutional layers to determine the output size
        x = self.conv_layer(x)
        return x.view(x.size(0), -1).size(1)

    def forward(self, x):
        # Define the forward pass through the network
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x


# Load the saved model from the path files generated from main_model.py, variant1_model.py, and variant2_model.py respectively
# TODO ------------------------------------------------------------------------------------------
# Uncomment current model
model_path = 'best_facial_expression_model.pth' # Load the Main Model
# model_path = 'path_variant1.pth' # Load Variant 1 model 
# model_path = 'path_variant2.pth' # Load Variant 2 model

# TODO ------------------------------------------------------------------------------------------
# Uncomment model descirption associated with current model being evaluated
# Model description in evaluation must correspond to the model used during training which generated the path file being interpretted in this script
model = OptimizedCNN()
# model = Variant1CNN()
# model = Variant2CNN()

# Load the state disctionary of the model from the specified path
model.load_state_dict(torch.load(model_path))

# Define the transformation to be applied to the images of the testing set currently evaluated
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
])

# Load the test dataset
# Modify path to the dataset currently tested to which transformations should be applied
test_dataset = ImageFolder(root='/Users/houry/OneDrive/Documents/CONCORDIA/SUMMER2024/COMP472/AIProject/COMP472_AI/data/labeled/', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Make predictions on the test data
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.append(predicted.cpu().numpy()) # Store predictions 
        all_labels.append(labels.cpu().numpy()) # Store true labels

# Concatenate the lists of predictions and labels into arrays to form a confusion matrix
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# Generate the confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Calculate true positive, true negative, false positive, and false negative
TP = cm[0, 0]
TN = cm[1, 1]
FP = cm[1, 0]
FN = cm[0, 1]

# Plot the confusion matrix
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks([0, 1, 2, 3], ['Angry', 'Focused', 'Happy', 'Neutral'])
plt.yticks([0, 1, 2, 3], ['Angry', 'Focused', 'Happy', 'Neutral'])
plt.text(0, 0, str(TP), ha="center", va="center", color="black", fontsize=18)
plt.text(1, 0, str(FP), ha="center", va="center", color="black", fontsize=18)
plt.text(0, 1, str(FN), ha="center", va="center", color="black", fontsize=18)
plt.text(1, 1, str(TN), ha="center", va="center", color="black", fontsize=18)
plt.text(2, 2, str(cm[2, 2]), ha="center", va="center", color="black", fontsize=18)
plt.text(3, 3, str(cm[3, 3]), ha="center", va="center", color="black", fontsize=18)
plt.text(0, 2, str(cm[0, 2]), ha="center", va="center", color="black", fontsize=18)
plt.text(0, 3, str(cm[0, 3]), ha="center", va="center", color="black", fontsize=18)
plt.text(1, 2, str(cm[1, 2]), ha="center", va="center", color="black", fontsize=18)
plt.text(1, 3, str(cm[1, 3]), ha="center", va="center", color="black", fontsize=18)
plt.text(2, 0, str(cm[2, 0]), ha="center", va="center", color="black", fontsize=18)
plt.text(2, 1, str(cm[2, 1]), ha="center", va="center", color="black", fontsize=18)
plt.text(3, 0, str(cm[3, 0]), ha="center", va="center", color="black", fontsize=18)
plt.text(3, 1, str(cm[3, 1]), ha="center", va="center", color="black", fontsize=18)
plt.text(2, 3, str(cm[2, 3]), ha="center", va="center", color="black", fontsize=18)
plt.text(3, 2, str(cm[3, 2]), ha="center", va="center", color="black", fontsize=18)

# TODO ------------------------------------------------------------------------------------------
# Uncomment title associated with current model
plt.suptitle("Main Model Confusion Matrix", fontsize=20, y=0.95) # Title for the Main Model confusion matrix
# plt.suptitle("Variant 1 Confusion Matrix", fontsize=20, y=0.95)  # Title for the Variant 1 confusion matrix
# plt.suptitle("Variant 2 Confusion Matrix", fontsize=20, y=0.95)  # Title for the Variant 2 confusion matrix

# Diplay the plot
plt.show()

# Print the confusion matrix
print(cm)
