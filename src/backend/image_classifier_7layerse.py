import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

torch.manual_seed(10)

model = models.resnet18(pretrained=True) #download pre-trained model

num_features = model.fc.in_features # check on layers from previous model

# class_names = {
#   "drivespeed?": [257, 360, 400, 450, 600],
#   "wheelsize?": [2, 2.75, 3.25, 4],
#   "size of robot? (a rough estimate of x, y)": None,  # Free response
#   "type of intake?": ["banded", "flexwheel"],
#   "what speed of intake?": None,  # Free response
#   "what tier hang letter? [a-h options]": ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
# }


#  Define loss function (CrossEntropyLoss with appropriate reduction)
criterion = torch.nn.BCELoss(reduction='mean')  # Binary cross-entropy with mean reduction --> defines predicted vs fthe actual --> it uses this equation: BCE = - (y * log(p) + (1 - y) * log(1 - p))
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)  # Adjust learning rate : makes it so it makes the difference as minimal as possible 


# Define your custom dataset class inheriting from Dataset
class RobotDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_paths = [os.path.join(data_dir, img) for img in os.listdir(data_dir) if img.endswith('.png')| img.endswith('.jpg')]  
        self.descriptions = ...  # Load descriptions from your pre-made data (dictionary or list of tuples)
        #others put self.fc1 and self.fc1 --> to assume inputs(images) and self.fc2 as preassigned tensor sizes but here we just define our variables straight forward
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = os.open(image_path).convert('RGB')  # Ensure RGB format
        if self.transform:
            image = self.transform(image) #transforms to tensor
        description_labels = self.descriptions[idx]  # Access labels based on index (modify for 7 binary labels)
        return image, description_labels  # Return image and corresponding labels

# Define transformations (resize, normalize)
# !!!!!! Tune this for the model you want
transform = transforms.Compose([
    transforms.Resize(224),  # Adjust as needed for your model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Freeze pre-trained layers
for param in model.parameters():
    param.requires_grad = False

# Define seven independent binary classification layers
model.fc = torch.nn.ModuleList([
    torch.nn.Sequential(
        torch.nn.Linear(num_features, 2),
        torch.nn.Sigmoid()
    ) for _ in range(7)  # Create 7 layers for 7 categories
])
num_epochs = 100 # change on how much data you have to start off with 

def train_model(model, criterion, optimizer, num_epochs, train_loader, val_loader, device):
    # Train the model for num_epochs
    best_loss = float('inf')
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        val_loss, val_acc = evaluate_model(model, criterion, val_loader, device)

        if val_loss < best_loss:
           best_loss = val_loss 
           #here we check with our evalution_model so the model being made doesnt overfitt to the dataset we give them

        # Train and validate the model on each epoch 
        # Use train and validate functions (implemented below)

# Training function:

# Training loop
def train(model, criterion, optimizer, train_loader, device):
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_corrects = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass, track history if only necessary
        with torch.set_grad_enabled(True):
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()


# Validation function:
def evaluate_model(model, criterion, val_loader, device):
    """
    Evaluates the model on the validation dataset.

    Args:
        model: Trained PyTorch model.
        criterion: Loss function (e.g., BCEWithLogitsLoss).
        val_loader: Data loader for the validation set.
        device: Device (CPU or GPU).

    Returns:
        A tuple containing average loss and accuracy on the validation set.
    """

    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():  # Disable gradient calculation for validation
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)  # Calculate loss

            # Update statistics
            running_loss += loss.item() * images.size(0)  # Batch size correction
            running_corrects += torch.sum(torch.round(outputs.data) == labels.data).item()

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = running_corrects / len(val_loader.dataset)

    print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return epoch_loss, epoch_acc
# Transfers images and labels to the device.
# Performs a forward pass to get model predictions.
# Calculates the loss using the provided criterion.
# Updates running loss and correct predictions.