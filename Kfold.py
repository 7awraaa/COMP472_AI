'''
This code took the main model from main_model.py.
Therefore the architecture of the model, as well as early stoppign and everything but the Kfolds were from main_models.py

These are the main resources from main_model.py that were used to build the architecture of the CNN model:
    "ChatGPT was used to ask how to improve accuracy based on this code and fix some errors we had that we did not understand, such as when the input convolutional layer did not match the output. 
    OpenAI, "ChatGPT: Chat Generative Pre-trained Transformer," OpenAI, San Francisco, CA, 2024. Available: https://chat.openai.com/. [Accessed: June 11, 2024].
    
    The blog post by Analytics Vidhya was used to understand the theory and implementation of CNNs in PyTorch which helped us gain further insights to be able to change the codes provided in the lab exercises. 
    Analytics Vidhya, "Building Image Classification Models Using CNN in PyTorch," 2019. Available: https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/. [Accessed: June 11, 2024].

    This YouTube video to gain further insights into CNNs and how they are constructed and used. 
    "Understanding Convolutional Neural Networks (CNNs) for Visual Recognition," YouTube. Available: https://www.youtube.com/watch?v=N_W4EYtsa10. [Accessed: June 11, 2024].

    This code was written by heavily refering to lab exercises 6 and 7 provided as course material from Concordia University in Montreal for the class COMP 472.
    Concordia University, "Lab Exercise 6," COMP 472, Montreal, QC, 2024. [Accessed: June 11, 2024].
    Concordia University, "Lab Exercise 7," COMP 472, Montreal, QC, 2024. [Accessed: June 11, 2024]."

    see below section for Kfold

'''
import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torchvision.models import resnet18
import numpy as np

#start time 
start_time_ev = time.time()

##Create a sequence of transformations to apply to the images.
#This includes converting images to tensors, normalizing them, and applying random color jitter for data augmentation.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
])

# Load dataset
# CHANGE PATH WHEN RUNNING CODE
dataset = ImageFolder(root='../data/labeled/', transform=transform)

# Define the CNN model exact same as main model copied form main_model.py
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
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_input_size = self._get_conv_output_size(torch.randn(1, 3, 256, 256))
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

# Training and validation function
#train_and_validate function takes several parameters: 
#train_loader and val_loader (data loaders for training and validation), model (neural network model), criterion (loss function), optimizer (optimizer for updating model parameters), scheduler (scheduler for adjusting learning rate), early_stopping (an instance of EarlyStopping class), and num_epochs (number of epochs for training, default 100)
def train_and_validate(train_loader, val_loader, model, criterion, optimizer, scheduler, early_stopping, num_epochs=100):
    kfold_model= copy.deepcopy(model.state_dict()) #This will hold the Kfold model state based on validation performance
    for epoch in range(num_epochs):  #iterates over the specified number of epochs
        model.train() #sets the model to training mode
        for images, labels in train_loader:
            outputs = model(images) #computes predictions (outputs) from the model
            loss = criterion(outputs, labels) #calculates the loss between predicted outputs and actual labels.
            optimizer.zero_grad() #clears previous gradients
            loss.backward() # computes gradients of loss 
            optimizer.step() #Updates model parameters based on gradients

        model.eval() #sets the model to evaluation mode (disables dropout and batch normalization).
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images) #computes predictions (outputs) from the model
                loss = criterion(outputs, labels) #calculates the loss between predicted outputs and actual labels
                val_loss += loss.item() #accumulates validation loss
                _, predicted = torch.max(outputs.data, 1) #predicted labels
                total += labels.size(0) #total number of labels.
                correct += (predicted == labels).sum().item() #correct predictions

        val_loss /= len(val_loader) #computes average validation loss
        val_accuracy = (correct / total) * 100 #computes validation accuracy
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
        scheduler.step(val_loss) #adjusts learning rate based on validation loss

        if val_loss < early_stopping.best_loss:
            kfold_model = copy.deepcopy(model.state_dict())
        
        early_stopping(val_loss) #checks for early stopping
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return kfold_model

# Early stopping class
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

'''
The referecnes below were used to be able to implement the k fold to our already exisiting model:

This article was used to understand the concept and implementation of Stratified K-Fold Cross-Validation, which ensures that each fold has a proportional representation of each class, thereby reducing bias in our model evaluation.
GeeksforGeeks, "Stratified K-Fold Cross-Validation," GeeksforGeeks, 2024 Available: https://www.geeksforgeeks.org/stratified-k-fold-cross-validation/. [Accessed: June 23, 2024].

The scikit-learn documentation was used as a reference to correctly implement the StratifiedKFold method in our model, ensuring that the cross-validation process was accurately conducted using the library's functionalities.
scikit-learn, "sklearn.model_selection.StratifiedKFold," scikit-learn, 2024. Available: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html. [Accessed: June 23, 2024].

This Medium article provided a comprehensive overview of different cross-validation techniques, including K-Fold Cross-Validation, which was crucial in enhancing our understanding of model evaluation and ensuring robustness in our project.
A. Jain, "Understanding Cross-Validation: Enhancing Model Evaluation," Medium, 2024.  Available: https://medium.com/@abhishekjainindore24/understanding-cross-validation-enhancing-model-evaluation-ccad3e19cde0. [Accessed: June 22, 2024].

The YouTube video was used to gain visual and practical insights into the cross-validation process, which helped us better understand how to implement K-Fold Cross-Validation in our model.
"Understanding Cross-Validation in Machine Learning," YouTube, 2024. Available: https://youtu.be/6dDet0-Drzc. [Accessed: June 24, 2024].

ChatGPT was used to ask how to apply K-Fold Cross-Validation to our existing model. It provided a comprehensive overview of the necessary steps, including defining the training and validation functions, data splitting, and integrating the StratifiedKFold method, which we then adjusted to fit our model.
OpenAI, "ChatGPT: Chat Generative Pre-trained Transformer," OpenAI, San Francisco, CA, 2024. Available: https://chat.openai.com/. [Accessed: June 23, 2024].
'''
# K-fold Cross-validation
k_folds = 10 #define the number of folds for cross-validation
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42) #initializes StratifiedKFold for splitting dataset into k_folds
results = []

#iterates through each fold, yielding train (train_idx) and test (test_idx) indices
#splits dataset into train and test subsets based on indices
for fold, (train_idx, test_idx) in enumerate(skf.split(dataset.samples, dataset.targets)):
    print(f'Fold {fold + 1}/{k_folds}')

    train_sub_idx, val_sub_idx = train_test_split(train_idx, test_size=0.1765, random_state=42)
    train_subset = Subset(dataset, train_sub_idx)
    val_subset = Subset(dataset, val_sub_idx)
    test_subset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

    model = OptimizedCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    early_stopping = EarlyStopping(patience=10, min_delta=0.01)

    #call train_and_validate function to train and validate the model on current fold's train and validation data
    kfold_model = train_and_validate(train_loader, val_loader, model, criterion, optimizer, scheduler, early_stopping)
    torch.save(kfold_model, f'k_model_fold_{fold + 1}.pth') #saves model state for current fold 
    model.load_state_dict(kfold_model) #loads the best model state for evaluation
    model.eval() #set model to evaluation mode

     #collect true labels and predicted labels respectively for the test set
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.numpy())

    '''
    As per the main_model.py this reference was used to compute the metrics
    
    "For understanding the evaluation metrics (accuracy, precision, recall, F1 score) and how to implement them in Python:
    Scikit-learn documentation: "sklearn.metrics.precision_score," . Available: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html. [Accessed: June 11, 2024]. "
    
    '''

    #computes metrics 
    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    precision_micro = precision_score(all_labels, all_preds, average='micro')
    recall_micro = recall_score(all_labels, all_preds, average='micro')
    f1_micro = f1_score(all_labels, all_preds, average='micro')

    results.append({
        'Fold': fold + 1,
        'Accuracy': accuracy,
        'Precision Macro': precision_macro,
        'Recall Macro': recall_macro,
        'F1 Macro': f1_macro,
        'Precision Micro': precision_micro,
        'Recall Micro': recall_micro,
        'F1 Micro': f1_micro
    })

# Print results for each fold
for result in results:
    print(f"Fold {result['Fold']}:")
    print(f"Accuracy: {result['Accuracy']:.4f}")
    print(f"Precision Macro: {result['Precision Macro']:.4f}")
    print(f"Recall Macro: {result['Recall Macro']:.4f}")
    print(f"F1 Macro: {result['F1 Macro']:.4f}")
    print(f"Precision Micro: {result['Precision Micro']:.4f}")
    print(f"Recall Micro: {result['Recall Micro']:.4f}")
    print(f"F1 Micro: {result['F1 Micro']:.4f}")
    print()

# Print average results
avg_accuracy = np.mean([result['Accuracy'] for result in results])
avg_precision_macro = np.mean([result['Precision Macro'] for result in results])
avg_recall_macro = np.mean([result['Recall Macro'] for result in results])
avg_f1_macro = np.mean([result['F1 Macro'] for result in results])
avg_precision_micro = np.mean([result['Precision Micro'] for result in results])
avg_recall_micro = np.mean([result['Recall Micro'] for result in results])
avg_f1_micro = np.mean([result['F1 Micro'] for result in results])

print(f'Average Accuracy: {avg_accuracy:.4f}')
print(f'Average Precision Macro: {avg_precision_macro:.4f}')
print(f'Average Recall Macro: {avg_recall_macro:.4f}')
print(f'Average F1 Macro: {avg_f1_macro:.4f}')
print(f'Average Precision Micro: {avg_precision_micro:.4f}')
print(f'Average Recall Micro: {avg_recall_micro:.4f}')
print(f'Average F1 Micro: {avg_f1_micro:.4f}')

end_time_ev = time.time()
print(f"Evaluation Time: {end_time_ev - start_time_ev} seconds")
