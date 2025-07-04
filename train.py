import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from underthesea import word_tokenize
import os
import json
# import logging

# # Configure logging
# logging.basicConfig(level=print, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-3
EMBEDDING_DIM = 256
NUM_FILTERS = 100
KERNEL_SIZES = [3, 4, 5]
DROPOUT_RATE = 0.5
MIN_WORD_FREQ = 2

# Paths for saving/loading
MODEL_PATH = "viet_toxic_comment_model_no_transformers.pth"
VOCAB_PATH = "custom_vocab.json"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Loading & Preprocessing ---
def load_and_preprocess_data(file_path):
    """
    Loads the CSV dataset and preprocesses labels.
    Assumes 'toxic' label is 1, others are 0.
    """
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        df['label'] = df['label'].apply(lambda x: 1 if x == 'toxic' else 0)
        df.dropna(subset=['comment'], inplace=True)
        print("Data loaded and labels preprocessed.")
        print(f"Dataset shape: {df.shape}")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        return df
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {file_path}")
        return None
    except KeyError as e:
        print(f"Error: Missing expected column in CSV: {e}. Please check your CSV file headers.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return None

# --- Custom Tokenization and Vocabulary Building ---
class CustomTokenizer:
    def __init__(self):
        self.word_to_idx = {"<pad>": 0, "<unk>": 1}
        self.idx_to_word = {0: "<pad>", 1: "<unk>"}
        self.vocab_size = 2

    def build_vocab(self, texts, min_freq=MIN_WORD_FREQ):
        print("Building custom vocabulary...")
        word_counts = {}
        for text in texts:
            words = word_tokenize(text.lower(), format="text").split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        for word, count in word_counts.items():
            if count >= min_freq:
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = self.vocab_size
                    self.idx_to_word[self.vocab_size] = word
                    self.vocab_size += 1
        print(f"Vocabulary built with {self.vocab_size} unique words.")

    def encode(self, text, max_len):
        words = word_tokenize(text.lower(), format="text").split()
        token_ids = [self.word_to_idx.get(word, self.word_to_idx["<unk>"]) for word in words]
        
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        else:
            padding_length = max_len - len(token_ids)
            token_ids = token_ids + [self.word_to_idx["<pad>"]] * padding_length
            
        attention_mask = [1] * len(words) + [0] * padding_length
        if len(attention_mask) > max_len:
             attention_mask = attention_mask[:max_len]

        return {"input_ids": token_ids, "attention_mask": attention_mask}

    def save_vocab(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.word_to_idx, f, ensure_ascii=False, indent=4)
        print(f"Vocabulary saved to {path}")

    @classmethod
    def load_vocab(cls, path):
        print(f"Loading vocabulary from {path}...")
        tokenizer = cls()
        with open(path, 'r', encoding='utf-8') as f:
            tokenizer.word_to_idx = json.load(f)
        tokenizer.idx_to_word = {v: k for k, v in tokenizer.word_to_idx.items()}
        tokenizer.vocab_size = len(tokenizer.word_to_idx)
        print(f"Vocabulary loaded with {tokenizer.vocab_size} words.")
        return tokenizer

# --- Custom Dataset for PyTorch ---
class CommentDataset(Dataset):
    def __init__(self, comments, labels, tokenizer, max_len):
        self.comments = comments
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        comment = str(self.comments[idx])
        label = self.labels[idx]
        encoded = self.tokenizer.encode(comment, self.max_len)
        
        return {
            'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# --- Model Definition (SimpleCNN) ---
class SimpleCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, kernel_sizes, dropout_rate):
        super(SimpleCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) 
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=num_filters,
                      kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, 1)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        embedded = embedded.permute(0, 2, 1) 
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

# --- Training Function ---
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device):
    print("Starting model training...")
    model.train()
    best_val_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask).squeeze(1) 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct_predictions / total_samples
        print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Model saved to {MODEL_PATH} (best validation loss: {best_val_loss:.4f})")

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask).squeeze(1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    model.train()
    return avg_loss, accuracy

if __name__ == "__main__":
    df = load_and_preprocess_data('vne_dataset.csv')
    if df is None:
        print("Exiting due to data loading error.")
        exit()

    comments = df['comment'].tolist()
    labels = df['label'].tolist()

    custom_tokenizer = CustomTokenizer()
    if not os.path.exists(VOCAB_PATH):
        custom_tokenizer.build_vocab(comments)
        custom_tokenizer.save_vocab(VOCAB_PATH)
    else:
        custom_tokenizer = CustomTokenizer.load_vocab(VOCAB_PATH)

    X_train, X_val, y_train, y_val = train_test_split(
        comments, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_dataset = CommentDataset(X_train, y_train, custom_tokenizer, MAX_LEN)
    val_dataset = CommentDataset(X_val, y_val, custom_tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    VOCAB_SIZE = custom_tokenizer.vocab_size
    
    model = SimpleCNN(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_filters=NUM_FILTERS,
        kernel_sizes=KERNEL_SIZES,
        dropout_rate=DROPOUT_RATE
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    train_model(model, train_loader, val_loader, optimizer, criterion, EPOCHS, device)
    print("Training complete. Model and vocabulary saved.")
