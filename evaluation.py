#In this code the same transformtaions, data splitting and architecture as the same model are applied. 
#The difference is instead of training it, the model is being evaluated here by loading the different model state dictionaries that were store previously as path files


''' ChatGPT was used to ask how to improve accuracy based on this code and fix some errors we had that we did not understand, such as when the input convolutional layer did not match the output. 
    OpenAI, "ChatGPT: Chat Generative Pre-trained Transformer," OpenAI, San Francisco, CA, 2024. Available: https://chat.openai.com/. [Accessed: June 11, 2024].
    
    The blog post by Analytics Vidhya was used to understand the theory and implementation of CNNs in PyTorch which helped us gain further insights to be able to change the codes provided in the lab exercises. 
    Analytics Vidhya, "Building Image Classification Models Using CNN in PyTorch," 2019. Available: https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/. [Accessed: June 11, 2024].

    This YouTube video to gain further insights into CNNs and how they are constructed and used. 
    "Understanding Convolutional Neural Networks (CNNs) for Visual Recognition," YouTube. Available: https://www.youtube.com/watch?v=N_W4EYtsa10. [Accessed: June 11, 2024].

    This code was written by heavily refering to lab exercises 6 and 7 provided as course material from Concordia University in Montreal for the class COMP 472.
    Concordia University, "Lab Exercise 6," COMP 472, Montreal, QC, 2024. [Accessed: June 11, 2024].
    Concordia University, "Lab Exercise 7," COMP 472, Montreal, QC, 2024. [Accessed: June 11, 2024].
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Define transformation (same as training)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
])

# Load dataset (ensure to use the same root path)
dataset = ImageFolder(root='/Users/houry/OneDrive/Documents/CONCORDIA/SUMMER2024/COMP472/AIProject/COMP472_AI/data/labeled/', transform=transform)

# Split dataset (ensure same split logic as during training)
train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.15, random_state=42)
train_idx, val_idx = train_test_split(train_idx, test_size=0.1765, random_state=42)

train_dataset = torch.utils.data.Subset(dataset, train_idx)
val_dataset = torch.utils.data.Subset(dataset, val_idx)
test_dataset = torch.utils.data.Subset(dataset, test_idx)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define your CNN model
class OptimizedCNN(nn.Module):
    def __init__(self):
        super(OptimizedCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            #when evaluating Variant 2 the kernel size needs to be changed to 5 and the padding to 2
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

            # TO ADD when evaluating variant 1
            #nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            #nn.BatchNorm2d(1024), 
            #nn.ReLU(inplace=True), 
            #nn.MaxPool2d(kernel_size=2, stride=2)
            
        )
        self.fc_input_size = self._get_conv_output_size(torch.randn(1, 3, 256, 256))  # Update input size
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
        x = self.conv_layer(x)
        return x.view(x.size(0), -1).size(1)

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x
    
''' The code below, used for evaluating the model, was crafted by referring to several sources, including lab exercises. 
    These resources aided our comprehension of how the model assesses data, enabling us to effectively apply code snippets from websites to our models.

    J. Brownlee, "How to Evaluate the Performance of PyTorch Models," Machine Learning Mastery, 2019.  Available: https://machinelearningmastery.com/how-to-evaluate-the-performance-of-pytorch-models/. [Accessed: June 12, 2024].

    Stack Overflow discussions and code examples for evaluating models. Available: https://stackoverflow.com/questions/71534943/how-to-evaluate-a-trained-model-in-pytorch. [Accessed: June 12, 2024].

'''

# Load the trained model
#CHANGE THE PATH for the variants
model = OptimizedCNN()
model.load_state_dict(torch.load('best_facial_expression_model.pth'))
model.eval() #The model is set to evaluation mode using the eval() method. This disables certain layers like dropout and batch normalization that behave differently during training and inference

# Evaluate the model's performance on a given dataset loader
#uses data loader to iterate through set and make predictions
def evaluate_model(loader):
    #initialize to store the predicted labels and the true labels
    all_preds = []
    all_labels = []
    #context manager is used to disable gradient calculation, which is not needed during evaluation and saves memory
    with torch.no_grad():
        #A loop iterates over batches of images and their corresponding labels from the dataset loader
        for images, labels in loader:
            outputs = model(images) #The model makes predictions on the batch of images. The output is a tensor containing the raw scores for each class
            _, predicted = torch.max(outputs.data, 1) #get the index of the maximum score for each image, which corresponds to the predicted class label
            #convert to NumPy and add to list
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    '''
    For understanding the evaluation metrics (accuracy, precision, recall, F1 score) and how to implement them in Python:
    Scikit-learn documentation: "sklearn.metrics.precision_score," . Available: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html. [Accessed: June 11, 2024]. 
    '''
    #Calculate accuracy of prediction and other metrics with sickit learn
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy, precision, recall, f1

#call fucntion with the test dataset loader, and the returned performance metrics are stored in respective variables to be printed
test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(test_loader)
print(f'Test Accuracy: {test_accuracy:.2f}')
print(f'Test Precision: {test_precision:.2f}')
print(f'Test Recall: {test_recall:.2f}')
print(f'Test F1 Score: {test_f1:.2f}')

# Example usage for predicting an individual image
from torchvision.io import read_image

#function to predict class of given image
def predict_image(image_path):
    image = read_image(image_path)
    image = transform(image).unsqueeze(0)  #the image is transformed using the predefined transform function and an additional batch dimension is added 
    with torch.no_grad():
        outputs = model(image) #The model makes a prediction on the image. The output is a tensor containing the raw scores for each class
        _, predicted = torch.max(outputs.data, 1) #get the index of the maximum score, which corresponds to the predicted class label
        class_idx = predicted.item() #predicted class index is extracted from the tensor
    return dataset.classes #return the class label corresponding to the predicted index

#CHANGE PATH to specific image
image_path = '/Users/houry/OneDrive/Documents/CONCORDIA/SUMMER2024/COMP472/AIProject/COMP472_AI/data/labeled/angry_faces/image0000007.jpg'  # Replace with your image path
predicted_class = predict_image(image_path)
print(f'The predicted class for the image is: {predicted_class}')
