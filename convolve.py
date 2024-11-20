import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import cv2
import torch.backends.cudnn as cudnn

# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Rest of your ASLDataset class remains the same
class ASLDataset(Dataset):
    """
    Custom Dataset for loading ASL images.

    Expected directory structure:
    root_dir/
        class_1/
            img1.jpg
            img2.jpg
            ...
        class_2/
            img1.jpg
            img2.jpg
            ...
        ...

    Note: You can combine datasets by adding photos from different datasets into the respective class folders.
    The file names can be anything as long as they are in the correct subfolder.
    """
    def __init__(self, root_dir, transform=None, preprocessing=None):
        self.root_dir = root_dir
        self.transform = transform
        self.preprocessing = preprocessing
        self.classes = sorted([d for d in os.listdir(root_dir) 
                             if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.preprocessing:
            image = self.preprocessing(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class TrainingProgress:
    def __init__(self):
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
    def update(self, train_loss, train_acc, val_loss, val_acc):
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)
        
    def plot_progress(self):
        epochs = range(1, len(self.train_losses) + 1)
        
        # Create a figure with two subplots
        plt.figure(figsize=(15, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        plt.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()
    pass

def evaluate_model(model, test_loader, criterion, device, classes):
    if len(test_loader.dataset) == 0:
        print("Test dataset is empty. Please check the test dataset and DataLoader.")
        return 0.0, 0.0
    
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    print(f"Number of test samples: {len(test_loader.dataset)}")  # Check the size of the test dataset
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    if len(all_preds) == 0 or len(all_labels) == 0:
        print("No predictions were made. Please check the test dataset and DataLoader.")
        return test_loss / len(test_loader), 0.0
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, 
                                 target_names=classes, digits=3)
    
    # Save report to file
    with open('classification_report.txt', 'w') as f:
        f.write(report)
    
    return test_loss / len(test_loader), 100 * correct / total

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                scheduler, num_epochs, device):
    # Add GPU memory optimization
    torch.cuda.empty_cache()
    
    progress = TrainingProgress()
    best_val_acc = 0.0
    scaler = torch.amp.GradScaler('cuda')  # Mixed precision scaler
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            with torch.amp.autocast('cuda'):  # Mixed precision training
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Update progress tracker
        progress.update(train_loss, train_acc, val_loss, val_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pth')
            print('Model saved!')
        
        # Plot progress after each epoch
        progress.plot_progress()
        print('-' * 60)
        pass

def main():
    # Check GPU availability and set device
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your PyTorch installation and GPU setup.")
        return
    
    device = torch.device("cuda:0")
    print(f"Using device: {device}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Create mixed precision scaler
    scaler = torch.amp.GradScaler('cuda')
    
    # Hyperparameters optimized for GPU
    batch_size = 64  # Increased batch size for GPU
    num_epochs = 4
    learning_rate = 0.001
    
    # Data augmentation and normalization for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Standard DenseNet input size
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transform
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Define any additional preprocessing steps
    preprocessing = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])
    
    # Update these paths to match your actual dataset location
    train_data_dir = "C:/Users/Prakarsh/Desktop/asl_dataset/train"
    test_data_dir = "C:/Users/Prakarsh/Desktop/asl_dataset/test"
    
    # Create dataset and dataloaders with GPU optimizations
    full_dataset = ASLDataset(train_data_dir, transform=train_transform, preprocessing=False)
    test_dataset = ASLDataset(test_data_dir, transform=val_transform, preprocessing=False)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size])
    
    # Use pin_memory=True for faster data transfer to GPU
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4,
                            pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=4,
                          pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4,
                           pin_memory=True, persistent_workers=True)
    
    # Load model and move to GPU
    model = models.densenet121(weights='IMAGENET1K_V1')  # Use pretrained weights
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, len(full_dataset.class_to_idx))
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, 
                scheduler, num_epochs, device)
    
    # Evaluate the model on the test dataset
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device, full_dataset.classes)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')

if __name__ == "__main__":
    main()