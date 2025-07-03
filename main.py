from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from flask import Flask, request, jsonify
import os


MAX_LEN = 128 # Example value, use the one you trained with

# label_map = {0: "Non-Toxic", 1: "Toxic"} # Example value, use the one you trained with

output_dir = "./model"


# Check if the saved model directory exists
if not os.path.exists(output_dir):
    print(f"Error: Saved model directory '{output_dir}' not found.")
    # You might want to exit or raise an error here
else:
    print(f"Loading model and tokenizer from {output_dir}...")
    try:


        # Load the tokenizer and model from the saved directory
        loaded_tokenizer = AutoTokenizer.from_pretrained("./saved_model", use_fast=False)
        loaded_model = AutoModelForSequenceClassification.from_pretrained("./saved_model")

        print("Tokenizer and model loaded successfully.")

        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loaded_model.to(device)
        print(f"Using device: {device}")

    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")

# --- Define the prediction function ---
def predict_toxicity(comment):

    if 'loaded_tokenizer' not in globals() or 'loaded_model' not in globals():
        return "Error: Model or tokenizer not loaded.", [0.0, 0.0]

    # Tokenize the input comment
    inputs = loaded_tokenizer(
        comment,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Move inputs to the appropriate device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Set model to evaluation mode
    loaded_model.eval()

    # Perform inference
    with torch.no_grad():
        outputs = loaded_model(**inputs)

    # Get logits
    logits = outputs.logits

    # Apply softmax to get probabilities
    probabilities = torch.softmax(logits, dim=1)

    # Determine the predicted class
    predictions = torch.argmax(probabilities, dim=1).cpu().item()

    # predicted_label = label_map[predictions]

    # Convert probabilities tensor to a Python list
    probabilities_list = probabilities.cpu().flatten().tolist()

    return predictions, probabilities_list

app = Flask(__name__)

print("Flask application initialized.")

@app.route('/predict', methods=['POST'])
def predict():
    if not request.json or 'comment' not in request.json:
        return jsonify({'error': 'Invalid request. Please provide a JSON object with a "comment" key.'}), 400

    comment = request.json['comment']

    predictions, probabilities = predict_toxicity(comment)

    if predictions.startswith("Error"):
         return jsonify({'error': predictions}), 500


    return jsonify({
        'predictions': predictions,
        'probabilities': probabilities
    })


app.run(debug=True, host='0.0.0.0', port=5000)