from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pickle
import os

# Init Flask
app = Flask(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model paths
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "nbs", "xgb_model.pkl")
encoder_path = os.path.join(base_dir, "nbs", "label_encoder.pkl")

# Load model and encoder
with open(model_path, "rb") as f:
    model = pickle.load(f)
with open(encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

# Load BERT model & tokenizer once
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased").to(device)
bert.eval()

# Convert input to BERT CLS embedding
def get_cls_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

# Webhook route for Dialogflow
@app.route("/webhook", methods=["POST"])
def webhook():
    req = request.get_json()
    user_input = req["queryResult"]["queryText"]

    # Vectorize and predict
    vector = get_cls_embedding(user_input)
    prediction = model.predict([vector])[0]
    intent = label_encoder.inverse_transform([prediction])[0]

    # Respond to Dialogflow
    return jsonify({
        "fulfillmentText": f"Predicted intent: {intent}"
    })

if __name__ == "__main__":
    app.run(port=5000)