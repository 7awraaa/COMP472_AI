import os
import time
import copy
import torch #PyTorch library for deep learning
import torch.nn as nn  #for neural network modules
import torch.optim as optim #for optimization algorithms
from torch.utils.data import DataLoader #for loading data
from torchvision import transforms #for image transformations
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score #for evaluating performance metrics
from sklearn.model_selection import train_test_split #for splitting datasets
from torchvision.models import resnet18 

start_time_ev = time.time()  #record the start time for evaluation

#Create a sequence of transformations to apply to the images.
#This includes converting images to tensors, normalizing them, and applying random color jitter for data augmentation.
transform = transforms.Compose([
    transforms.ToTensor(), #convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), #normalize
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5) #randomly change the brightness, contrast, saturation, and hue
])

# Load dataset and apply transformation to each image
#CHANGE PATH WHEN RUNNING CODE
dataset = ImageFolder(root='/Users/houry/OneDrive/Documents/CONCORDIA/SUMMER2024/COMP472/AIProject/COMP472_AI/data/labeled/', transform=transform)

# Split dataset into training, validation, and test sets
train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.15, random_state=42) #Split dataset into 85% train+val and 15% test
train_idx, val_idx = train_test_split(train_idx, test_size=0.1765, random_state=42) #Further split train+val into train and val (80% for training and 20% for validation)

train_dataset = torch.utils.data.Subset(dataset, train_idx) #subset for training data
val_dataset = torch.utils.data.Subset(dataset, val_idx) #subset for validation data
test_dataset = torch.utils.data.Subset(dataset, test_idx) #subset for test data

#Create Dataloaders for the different subsets of data where they all have a batch size of 32 but only the training data has shuffle true
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) #dataLoader for training
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) #dataLoader for validation
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  #dataLoader for testing

# Sample input for determining the fully connected layer size
sample_input = torch.randn(1, 3, 256, 256) #size of image 3 channels and 256x256 pixels
classes = ('angry_faces', 'focused_faces', 'happy_faces', 'neutral_faces')

# Define the CNN model
class OptimizedCNN(nn.Module):
    def __init__(self):
        super(OptimizedCNN, self).__init__()
        # Define convolutional layers
        self.conv_layer = nn.Sequential #combines multiple layers into a single sequential module. The input is passed through each layer in order
        ( 
            #perform 2D convolution operations
            #each layer has a specified number of input channels (in_channels), output channels (out_channels), kernel size (kernel_size), and padding (padding)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), #batch normalization, normalize the activations of the previous layer to improve training stability and performance
            nn.ReLU(inplace=True),  #ReLU activation, introduce non-linearity to the model. The inplace=True parameter allows for in-place operation, saving memory
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), #batch normalization
            nn.ReLU(inplace=True), #ReLU activation
            nn.MaxPool2d(kernel_size=2, stride=2), #max pooling, reduces the spatial dimensions (height and width) of the input
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), #batch normalization
            nn.ReLU(inplace=True), #ReLU activation
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), #batch normalization
            nn.ReLU(inplace=True),  #ReLU activation
            nn.MaxPool2d(kernel_size=2, stride=2), #max pooling
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  #batch normalization
            nn.ReLU(inplace=True),  #ReLU activation
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), #batch normalization
            nn.ReLU(inplace=True), #ReLU activation
            nn.MaxPool2d(kernel_size=2, stride=2), #max pooling

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), #batch normalization
            nn.ReLU(inplace=True), #ReLU activation
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), #batch normalization
            nn.ReLU(inplace=True), #ReLU activation
            nn.MaxPool2d(kernel_size=2, stride=2), #max pooling

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), #batch normalization
            nn.ReLU(inplace=True), #ReLU activation
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), #batch normalization
            nn.ReLU(inplace=True), #ReLU activation
            nn.MaxPool2d(kernel_size=2, stride=2), #max pooling

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),  # New convolutional layer
            nn.BatchNorm2d(512), #batch normalization
            nn.ReLU(inplace=True), #ReLU activation
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), #batch normalization
            nn.ReLU(inplace=True), #ReLU activation
            nn.MaxPool2d(kernel_size=2, stride=2), #max pooling

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024), #batch normalization
            nn.ReLU(inplace=True), #ReLU activation
            nn.MaxPool2d(kernel_size=2, stride=2) #max pooling
            
        )
        #Fully connected layers
        self.fc_input_size = self._get_conv_output_size(sample_input) #Calculate the input size for the fully connected layer based on the output size of the convolutional layers
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.5), #dropout for regularization, randomly zeroes some of the elements of the input tensor with probability p (0.5 here) during training
            nn.Linear(self.fc_input_size, 1024),  #fully connected layer 1, connects the flattened output of the convolutional layers to 1024 neurons
            nn.ReLU(inplace=True), # ReLU activation
            nn.Linear(1024, 512), #fully connected layer 2, connects these 1024 neurons to 512 neurons
            nn.ReLU(inplace=True), # ReLU activation function to the outputs of the fully connected layers
            nn.Dropout(p=0.5), #dropout for regularization
            nn.Linear(512, 4) #output layer, connects 512 neurons to 4 output neurons (one for each class)
        )

    #get the output size for the fully connected layer
    def _get_conv_output_size(self, x):
        x = self.conv_layer(x)
        return x.view(x.size(0), -1).size(1)
    #forward pass to apply the convolutional and fully connected layers to the input
    def forward(self, x):
        x = self.conv_layer(x) #apply convolutional layers
        x = x.view(x.size(0), -1) #flatten the output
        x = self.fc_layer(x) #apply fully connected layers
        return x


# Initialize model, criterion, and optimizer
model = OptimizedCNN()
criterion = nn.CrossEntropyLoss() #loss function for classification
optimizer = optim.Adam(model.parameters(), lr=0.0001) #Adam optimizer with a learning rate of 0.0001

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
early_stopping = EarlyStopping(patience=10, min_delta=0.01)  #instantiate the early stopping object with specific parameters
best_model = copy.deepcopy(model.state_dict()) #create a deep copy of the model's state dictionary to store the best model

# Training loop
num_epochs = 100 #set the number of epochs to train, was set as high number to let the model run as logn as needed

#loop over the number of epochs
for epoch in range(num_epochs):
    model.train() #set the model to training mode
    for images, labels in train_loader: #loop over the batches of training data
        outputs = model(images) #get the model's predictions.
        loss = criterion(outputs, labels) #calculate the loss
        optimizer.zero_grad() #zero the parameter gradients
        loss.backward() #backpropagation
        optimizer.step() #update weights

    model.eval()  #set the model to evaluation mode
    val_loss = 0 #initialize the validation loss.
    correct = 0 #initialize counters for correct predictions 
    total = 0 #initialize counter for total samples

    with torch.no_grad(): #disable gradient calculation for validation

        #loop over the batches of validation data
        for images, labels in val_loader:
            outputs = model(images) #get the model's predictions
            loss = criterion(outputs, labels) #calculate the validation loss
            val_loss += loss.item() #Accumulate the validation loss
            _, predicted = torch.max(outputs.data, 1) #get the predicted classes
            total += labels.size(0) #update the total samples count
            correct += (predicted == labels).sum().item() #update the correct predictions count

    val_loss /= len(val_loader) #calculate the average validation loss
    val_accuracy = (correct / total) * 100 #Calculate the validation accuracy
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%') #print the validation loss and accuracy

    scheduler.step(val_loss) #adjust learning rate based on validation loss

    #check if the current validation loss is the best so far
    if val_loss < early_stopping.best_loss:
        best_model = copy.deepcopy(model.state_dict()) #save the current model state as the best model

    early_stopping(val_loss) #update the early stopping mechanism
    #check if early stopping is triggered
    if early_stopping.early_stop:
        print("Early stopping")
        break #exit the training loop

#save the best model's state dictionary to a file
torch.save(best_model, 'best_facial_expression_model.pth')

# Evaluate the model on the test set
model.load_state_dict(best_model) #load the best model's state
model.eval() #set the model to evaluation mode
#initialize lists to store all predictions and labels
all_preds = [] 
all_labels = []
#disable gradient calculation for testing
with torch.no_grad():
    #initialize counters for correct predictions and total samples
    correct = 0
    total = 0

    #loop over the batches of test data
    for images, labels in test_loader:
        outputs = model(images) #get the model's predictions
        _, predicted = torch.max(outputs.data, 1) #get the predicted classes
        total += labels.size(0) #update the total samples count
        correct += (predicted == labels).sum().item() #update the correct predictions count
        all_preds.extend(predicted.cpu().numpy()) #store the predictions
        all_labels.extend(labels.cpu().numpy()) #store the labels

    #calculate the test accuracy and print
    test_accuracy = (correct / total) * 100
    print(f'Test Accuracy: {test_accuracy:.2f}%')

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds) #calculate the accuracy
precision = precision_score(all_labels, all_preds, average='weighted') #calculate the weighted precision
recall = recall_score(all_labels, all_preds, average='weighted') #calculate the weighted recall
f1 = f1_score(all_labels, all_preds, average='weighted') #calculate the weighted F1 score
#Print them
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

end_time_ev = time.time() #record the end time for evaluation
#calculate the elapsed time and print
print(f"Evaluation running time: {(end_time_ev - start_time_ev)/60:.2f} minutes")