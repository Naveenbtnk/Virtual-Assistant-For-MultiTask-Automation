import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# Load intents
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Prepare data
all_words = []
tags = []
xy = []

# Tokenize and gather all patterns and tags
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        words = tokenize(pattern)
        all_words.extend(words)
        xy.append((words, tag))

# Stem and lower words, ignoring some symbols
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    y_train.append(tags.index(tag))

X_train = np.array(X_train)
y_train = np.array(y_train)

# Model parameters
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000
batch_size = 8
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import json

# Load the intents JSON file
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Extract tags and patterns
tags = []
patterns = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.extend([tag] * len(intent['patterns']))  # Repeat the tag for each pattern
    for pattern in intent['patterns']:
        patterns.append(pattern)

# Create a DataFrame for easy analysis
df = pd.DataFrame({
    'tag': tags,
    'patterns': patterns,
})

# 1. Distribution of Tags
plt.figure(figsize=(10, 6))
tag_counts = df['tag'].value_counts()
sns.barplot(x=tag_counts.index, y=tag_counts.values)
plt.title('Distribution of Tags')
plt.savefig('Distribution of Tags.png')
plt.xlabel('Tag')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

# 2. Word Frequency across all patterns
all_words = ' '.join(patterns).lower().split()
word_counts = Counter(all_words)
most_common_words = word_counts.most_common(20)  # Top 20 most common words
words, counts = zip(*most_common_words)

plt.figure(figsize=(10, 6))
sns.barplot(x=list(words), y=list(counts))
plt.title('Top 20 Most Common Words')
plt.savefig('Top 20 Most Common Words.png')
plt.xlabel('Word')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

# 3. Distribution of Pattern Lengths
pattern_lengths = [len(pattern.split()) for pattern in patterns]

plt.figure(figsize=(10, 6))
sns.histplot(pattern_lengths, kde=True)
plt.title('Distribution of Pattern Lengths')
plt.savefig('Distribution of Pattern Lengths.png')
plt.xlabel('Pattern Length (number of words)')
plt.ylabel('Frequency')
plt.show()

# Custom Dataset
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


# DataLoader
dataset = ChatDataset()
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model, Loss, Optimizer
model = NeuralNet(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Collect metrics and loss values for visualization
loss_values = []
accuracy_values = []
precision_values = []
recall_values = []
f1_values = []
# Train the model
for epoch in range(num_epochs):
    all_preds = []
    all_labels = []
    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Collect predictions and labels for metrics
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # Store the metrics for plotting
    loss_values.append(loss.item())
    accuracy_values.append(accuracy)
    precision_values.append(precision)
    recall_values.append(recall)
    f1_values.append(f1)

    # Print metrics every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')

# Save the trained model
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags,
}

FILE = "data1.pth"
torch.save(data, FILE)

print(f'Training complete. File saved to {FILE}')

# Plot the loss, accuracy, precision, recall, and F1-score over epochs

# Plot Loss
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), loss_values)
plt.title('Training Loss Over Epochs')
plt.savefig('Training_Loss_Over_Epochs.png')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot Accuracy
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), accuracy_values)
plt.title('Training Accuracy Over Epochs')
plt.savefig('Training Accuracy Over Epochs.png')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Plot Precision
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), precision_values)
plt.title('Training Precision Over Epochs')
plt.savefig('Training Precision Over Epochs.png')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.show()

# Plot Recall
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), recall_values)
plt.title('Training Recall Over Epochs')
plt.savefig('Training Recall Over Epochs.png')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.show()

# Plot F1-Score
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), f1_values)
plt.title('Training F1-Score Over Epochs')
plt.savefig('Training F1-Score Over Epochs.png')
plt.xlabel('Epoch')
plt.ylabel('F1-Score')
plt.show()
