import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
import json
import logging
from underthesea import word_tokenize # For Vietnamese word segmentation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration (must match training config) ---
MAX_LEN = 128
EMBEDDING_DIM = 256
NUM_FILTERS = 100
KERNEL_SIZES = [3, 4, 5]
DROPOUT_RATE = 0.5 # Dropout is typically off during inference, but model expects it during init

# Paths for loading saved assets
MODEL_PATH = "viet_toxic_comment_model_no_transformers.pth"
VOCAB_PATH = "custom_vocab.json"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# --- Custom Tokenization and Vocabulary Class (replicated for standalone app.py) ---
class CustomTokenizer:
    def __init__(self):
        self.word_to_idx = {"<pad>": 0, "<unk>": 1}
        self.idx_to_word = {0: "<pad>", 1: "<unk>"}
        self.vocab_size = 2

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

    @classmethod
    def load_vocab(cls, path):
        logging.info(f"Loading vocabulary from {path}...")
        tokenizer = cls()
        try:
            with open(path, 'r', encoding='utf-8') as f:
                tokenizer.word_to_idx = json.load(f)
            tokenizer.idx_to_word = {v: k for k, v in tokenizer.word_to_idx.items()}
            tokenizer.vocab_size = len(tokenizer.word_to_idx)
            logging.info(f"Vocabulary loaded with {tokenizer.vocab_size} words.")
            return tokenizer
        except FileNotFoundError:
            logging.error(f"Vocabulary file not found at {path}. Please run train_model.py first.")
            return None
        except Exception as e:
            logging.error(f"Error loading vocabulary: {e}")
            return None

# --- Model Definition (SimpleCNN - replicated for standalone app.py) ---
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

# --- Flask Backend Service ---
app = Flask(__name__)

# Global variables for loaded model and tokenizer
inference_custom_tokenizer = None
inference_model = None

try:

    logging.info("Loading assets for Flask inference...")
        
    inference_custom_tokenizer = CustomTokenizer.load_vocab(VOCAB_PATH)
    if inference_custom_tokenizer is None:
        logging.error("Failed to load custom tokenizer. Inference will not work.")
     

    inference_model = SimpleCNN(
        vocab_size=inference_custom_tokenizer.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_filters=NUM_FILTERS,
        kernel_sizes=KERNEL_SIZES,
        dropout_rate=DROPOUT_RATE
    ).to(device)


    inference_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    inference_model.eval() # Set model to evaluation mode
    logging.info("Model loaded successfully for Flask inference.")
except FileNotFoundError:
    logging.error(f"Error: Model file not found at {MODEL_PATH}. Please run train_model.py first.")
    inference_model = None
except Exception as e:
    logging.error(f"An error occurred loading the model: {e}")
    inference_model = None



@app.route('/predict', methods=['POST'])
def predict():
    if inference_custom_tokenizer is None or inference_model is None:
        return jsonify({'error': 'Model or tokenizer not loaded. Check server logs.'}), 500

    data = request.get_json(force=True)
    comment = data.get('comment', '')

    if not comment:
        return jsonify({'error': 'No comment provided in the request.'}), 400

    try:
        encoded = inference_custom_tokenizer.encode(comment, MAX_LEN)
        
        input_ids_tensor = torch.tensor([encoded['input_ids']], dtype=torch.long).to(device)
        attention_mask_tensor = torch.tensor([encoded['attention_mask']], dtype=torch.long).to(device)

        with torch.no_grad():
            output = inference_model(input_ids_tensor, attention_mask_tensor).squeeze(1)
            probability = torch.sigmoid(output).item()

        prediction_label = "toxic" if probability > 0.5 else "non-toxic"

        return jsonify({
            'comment': comment,
            'prediction_probability': probability,
            'predicted_label': prediction_label
        })

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

if __name__ == '__main__':
    # To run the Flask app, use:
    app.run(host='0.0.0.0', port=5000, debug=True)
