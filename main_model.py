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
import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torchvision.models import resnet18 

start_time_ev = time.time() # record the start time for evaluation

#Create a sequence of transformations to apply to the images.
#This includes converting images to tensors, normalizing them, and applying random color jitter for data augmentation.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5) #### version 1.1
])

# Load dataset and apply transformation to each image
#CHANGE PATH WHEN RUNNING CODE
dataset = ImageFolder(root='/Users/houry/OneDrive/Documents/CONCORDIA/SUMMER2024/COMP472/AIProject/COMP472_AI/data/labeled/', transform=transform)

# Split dataset into training, validation, and test sets
train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.15, random_state=42)
train_idx, val_idx = train_test_split(train_idx, test_size=0.1765, random_state=42)

train_dataset = torch.utils.data.Subset(dataset, train_idx)
val_dataset = torch.utils.data.Subset(dataset, val_idx)
test_dataset = torch.utils.data.Subset(dataset, test_idx)

#Create Dataloaders for the different subsets of data where they all have a batch size of 32 but only the training data has shuffle true
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Sample input for determining the fully connected layer size
sample_input = torch.randn(1, 3, 256, 256)  # Update input size to 256x256
classes = ('angry_faces', 'focused_faces', 'happy_faces', 'neutral_faces')

# Define the CNN model
class OptimizedCNN(nn.Module):
    def __init__(self):
        super(OptimizedCNN, self).__init__()
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
        self.fc_input_size = self._get_conv_output_size(sample_input)
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

''' The code below, which encompasses learning rate scheduling, optimizers, and early stopping, was developed with insights from various resources to understand these concepts and their implementation. 
    Multiple resources, including ChatGPT for summarizing findings and lab exercises, were consulted. The code references the following websites:
    
    "ChatGPT: Chat Generative Pre-trained Transformer," OpenAI, San Francisco, CA, 2024. Available: https://chat.openai.com/. [Accessed: June 11, 2024].

    DebuggerCafe, "Using Learning Rate Scheduler and Early Stopping with PyTorch," 2021. Available: https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/. [Accessed: June 11, 2024].

    Stack Overflow, "Early stopping in PyTorch,". Available: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch. [Accessed: June 11, 2024].

    V. Bhatt, "A Step-by-Step Guide to Early Stopping in TensorFlow and PyTorch," Medium, 2022. Available: https://medium.com/@vrunda.bhattbhatt/a-step-by-step-guide-to-early-stopping-in-tensorflow-and-pytorch-59c1e3d0e376. [Accessed: June 11, 2024].
    
    PyTorch documentation for optimizers and vision transforms. Available: https://pytorch.org/docs/stable/optim.html. [Accessed: June 11, 2024]
    
    Kaggle notebook for practical implementation examples. Available: https://www.kaggle.com/code/yusufmuhammedraji/pytorch-cv-earlystopping-lrscheduler. [Accessed: June 11, 2024].
'''

# Initialize model, criterion, and optimizer
model = OptimizedCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#create a scheduler to reduce the learning rate when the validation loss plateaus.
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)


# early stopping class to stop training if validation loss doesn't improve
#avoid overfitting
class EarlyStopping:
     #initialize with patience and minimum delta.
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    #check if the validation loss has improved and update the counter
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

#Initialize early stopping
early_stopping = EarlyStopping(patience=10, min_delta=0.01)
best_model = copy.deepcopy(model.state_dict())

# Training loop
num_epochs = 100

#loop over the number of epochs
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader: # loops over the batches of training data
        outputs = model(images) # get the model's predictions
        loss = criterion(outputs, labels) # calculate the loss
        optimizer.zero_grad() # zero the parameter gradient for every epoch
        loss.backward() # backpropagation
        optimizer.step() # update weights

    model.eval()
    val_loss = 0 # initialize validation loss
    correct = 0 # initialize counters
    total = 0 # initialize counter for total samples

    with torch.no_grad():
        #loop over the batches of validation data
        for images, labels in val_loader:
            outputs = model(images) # get the model's predictions
            loss = criterion(outputs, labels) # calculate the validation loss
            val_loss += loss.item() # accumulate the validation loss
            _, predicted = torch.max(outputs.data, 1) # get the predicted classes
            total += labels.size(0) # update the total samples count
            correct += (predicted == labels).sum().item() # update the correct predictions count

    val_loss /= len(val_loader) # calculate avg validation loss
    val_accuracy = (correct / total) * 100 # calculate validation accuracy
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    scheduler.step(val_loss) # adjust learning rate based on validation loss

    # check if current validation loss is the best so far
    if val_loss < early_stopping.best_loss:
        best_model = copy.deepcopy(model.state_dict()) #save the current model state as the best model if best so far

    early_stopping(val_loss) # update early stopping mechanism
    #check if early stopping is triggered
    if early_stopping.early_stop:
        print("Early stopping")
        break #exit the training loop when early stopping is triggered

#save the best model's state dictionary to a file
torch.save(best_model, 'best_facial_expression_model.pth')

# Evaluate the model on the test set
model.load_state_dict(best_model) # load best model's state
model.eval() # set the model to evaluation mode
# initialize lists to store all predictions and labels
all_preds = []
all_labels = []
with torch.no_grad():
    # initialize counters for correct predictions and total samples
    correct = 0
    total = 0

    # loop over the batches of test data
    for images, labels in test_loader:
        outputs = model(images) # get the model's predictions
        _, predicted = torch.max(outputs.data, 1) # get the predicited class
        total += labels.size(0) # update the total samples count
        correct += (predicted == labels).sum().item() # update the correct predictions count
        all_preds.extend(predicted.cpu().numpy()) # store the predictions
        all_labels.extend(labels.cpu().numpy()) # store the labels

    # calculate the test accuracy
    test_accuracy = (correct / total) * 100
    print(f'Test Accuracy: {test_accuracy:.2f}%')

'''
For understanding the evaluation metrics (accuracy, precision, recall, F1 score) and how to implement them in Python:
Scikit-learn documentation: "sklearn.metrics.precision_score," . Available: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html. [Accessed: June 11, 2024]. 
'''
# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds) # calculate accuracy
precision = precision_score(all_labels, all_preds, average='weighted') # calculate weighted precision
recall = recall_score(all_labels, all_preds, average='weighted') # calculate weighted recall
f1 = f1_score(all_labels, all_preds, average='weighted') # calculate weighted F1 score
# Print metrics
print(f'Test Accuracy: {accuracy:.2f}')
print(f'Test Precision: {precision:.2f}')
print(f'Test Recall: {recall:.2f}')
print(f'Test F1 Score: {f1:.2f}')

# Calculate micro and macro precision, recall, and F1 scores
precision_micro = precision_score(all_labels, all_preds, average='micro')
recall_micro = recall_score(all_labels, all_preds, average='micro')
f1_micro = f1_score(all_labels, all_preds, average='micro')
precision_macro = precision_score(all_labels, all_preds, average='macro')
recall_macro = recall_score(all_labels, all_preds, average='macro')
f1_macro = f1_score(all_labels, all_preds, average='macro')

print(f'Precision (Micro): {precision_micro:.2f}')
print(f'Recall (Micro): {recall_micro:.2f}')
print(f'F1 Score (Micro): {f1_micro:.2f}')
print(f'Precision (Macro): {precision_macro:.2f}')
print(f'Recall (Macro): {recall_macro:.2f}')
print(f'F1 Score (Macro): {f1_macro:.2f}')

end_time_ev = time.time() # record the end time for evaluation
# calculate the elapsed time and print
print(f"Evaluation running time: {(end_time_ev - start_time_ev)/60:.2f} minutes")