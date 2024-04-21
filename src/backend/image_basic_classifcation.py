import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Set random seed for reproducibility
data_transforms
torch.manual_seed(10)

class_names = {
  "drivespeed?": [257, 360, 400, 450, 600],
  "wheelsize?": [2, 2.75, 3.25, 4],
  "size of robot? (a rough estimate of x, y)": None,  # Free response
  "type of intake?": ["banded", "flexwheel"],
  "what speed of intake?": None,  # Free response
  "what tier hang letter? [a-h options]": ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
}

#load in model here
model = models.resnet18(pretrained=True)
# Get number of features from the last layer
num_features = model.fc.in_features

# Define seven independent binary classification layers
model.fc = torch.nn.ModuleList([
    torch.nn.Sequential(
        torch.nn.Linear(num_features, 2),
        torch.nn.Sigmoid()
    ) for _ in range(7)  # Create 7 layers for 7 categories
])

# Define loss function (CrossEntropyLoss with appropriate reduction)
criterion = torch.nn.BCELoss(reduction='mean')  # Binary cross-entropy with mean reduction
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)  # Adjust learning rate

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3 * 256 * 256, 128)  # Assuming input images are RGB and 256x256
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1000, 8)  # Output layer with 2 classes  # ADJUST THIS 
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 3 * 256 * 256)  # Flatten the input images (reshapes the input tensor to have a size of (batch_size, 3 * 256 * 256)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]) # set up the data transformation

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.dataset = ImageFolder(root, transform=transform)  # for making images into tensor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

# Define data loaders
train_dataset = CustomDataset(root='Flowers_Classification_dataset/train', transform=transform) # for the datset I gain
test_dataset = CustomDataset(root='Flowers_Classification_dataset/val', transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# main main components

# Training the model
num_epochs = 50
train_losses = []
valid_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    average_train_loss = running_loss / len(train_loader)
    train_losses.append(average_train_loss)

    # Validation
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()

    average_valid_loss = valid_loss / len(test_loader)
    valid_losses.append(average_valid_loss)

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Training Loss: {average_train_loss:.4f}, '
          f'Validation Loss: {average_valid_loss:.4f}')

# Evaluate the model
def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

evaluate_model(model, test_loader)

# Visualize training and validation curves
# plt.plot(train_losses, label='Training Loss')
# plt.plot(valid_losses, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# Save the trained model
torch.save(model.state_dict(), 'Robot_classification.pth')
description = []
for key, value in class_names.items():
    if isinstance(value, list):  # Handle options (categorical)
      description.append(f"{key}: {value[predicted_class_idx]} (options: {', '.join(value)})\n")
    else:  # Handle free response
      description += f"{key}: (Free Response)\n"

  return description