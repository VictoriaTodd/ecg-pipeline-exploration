from cap_sleep_dataset_hrv import CapSleepDatasetHRV
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import glob
import os
from sklearn.model_selection import train_test_split
import torch.optim as optim
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from toy_cnn import ToyCNN

# Open a new file
file1 = open("cap_sleep_run_cnn_hrv.txt", "a")

all_files = glob.glob("cap-sleep-database-1.0.0/*.edf")
all_files.sort()

train_files, test_files = train_test_split(
    all_files, test_size=0.2, random_state=42
)

print("creating dataset")

train_dataset = CapSleepDatasetHRV(train_files)
test_dataset = CapSleepDatasetHRV(test_files)

print("creating dataloader")

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

model = ToyCNN()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device =  ", device)

from torchview import draw_graph

# Move model to device
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10

def train_epoch(dataloader, model, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate(dataloader, model):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    return accuracy, report

# Training loop
for epoch in range(num_epochs):
    train_loss = train_epoch(train_dataloader, model, criterion, optimizer)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}")

# Evaluation
accuracy, report = evaluate(test_dataloader, model)
print(f"Test Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)

file1.write("Test Accuracy: " + accuracy + '\n')
file1.write("Classification Report:" + '\n')
file1.write(report  + '\n')
