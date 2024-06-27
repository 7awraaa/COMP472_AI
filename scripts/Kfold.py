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
    kfold_model= copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = (correct / total) * 100
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
        scheduler.step(val_loss)

        if val_loss < early_stopping.best_loss:
            kfold_model = copy.deepcopy(model.state_dict())
        
        early_stopping(val_loss)
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


# K-fold Cross-validation
k_folds = 10
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
results = []

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

    kfold_model = train_and_validate(train_loader, val_loader, model, criterion, optimizer, scheduler, early_stopping)
    torch.save(kfold_model, f'k_model_fold_{fold + 1}.pth')
    model.load_state_dict(kfold_model)
    model.eval()

    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.numpy())

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
