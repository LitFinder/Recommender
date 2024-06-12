import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, models

# Load the dataset
df = pd.read_csv('data_book_metadata.csv')

# Extract the sentences from the Metadata column
sentences = df['Metadata'].tolist()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Tokenize the sentences without truncation to get their lengths
token_lengths = [len(tokenizer.encode(sentence, add_special_tokens=True)) for sentence in sentences]

# Calculate the average and maximum token lengths
average_token_length = np.mean(token_lengths)
max_token_length = np.max(token_lengths)

print(f'Average token length: {average_token_length}')
print(f'Maximum token length: {max_token_length}')

# Tokenize the sentences with truncation to max_length of 512
inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=512)

# Extract input_ids and attention_mask
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Define a custom dataset
class TextDataset(Dataset):
    def __init__(self, input_ids, attention_mask, targets):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.targets = targets

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'targets': self.targets[idx]
        }

# Generate random target embeddings for demonstration purposes (replace with actual targets)
target_embeddings = np.random.rand(len(sentences), 256).astype(np.float32)

# Split the data into training and validation sets
train_inputs, val_inputs, train_targets, val_targets = train_test_split(input_ids, target_embeddings, test_size=0.1, random_state=42)
train_masks, val_masks = train_test_split(attention_mask, test_size=0.1, random_state=42)

# Create PyTorch datasets
train_dataset = TextDataset(train_inputs, train_masks, train_targets)
val_dataset = TextDataset(val_inputs, val_masks, val_targets)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Load the pre-trained model
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Define the fine-tuning model
class FineTuningModel(nn.Module):
    def __init__(self, base_model):
        super(FineTuningModel, self).__init__()
        self.base_model = base_model
        self.dense = nn.Linear(384, 256)  # 384 is the hidden size of all-MiniLM-L6-v2

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0, :]
        dense_output = self.dense(cls_token)
        return dense_output

# Initialize the fine-tuning model
fine_tuned_model = FineTuningModel(model)

# Define optimizer and loss function
optimizer = optim.SGD(fine_tuned_model.parameters(), lr=1e-4, momentum=0.9)
criterion = nn.MSELoss()

# Training function
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# Evaluation function
def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
    return total_loss / len(dataloader)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fine_tuned_model.to(device)

epochs = 2
best_val_loss = float('inf')
early_stopping_patience = 5
early_stopping_counter = 0

for epoch in range(epochs):
    train_loss = train_epoch(fine_tuned_model, train_dataloader, optimizer, criterion, device)
    val_loss = evaluate_epoch(fine_tuned_model, val_dataloader, criterion, device)
    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        torch.save(fine_tuned_model.state_dict(), 'fine_tuned_all_mini_lm_l6_v2.pt')
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= early_stopping_patience:
        print("Early stopping")
        break

# Load the fine-tuned model for inference or further training
fine_tuned_model.load_state_dict(torch.load('fine_tuned_all_mini_lm_l6_v2.pt'))

fine_tuned_model.save('fine_tuned_all_mini_lm_l6_v2')
