# from flask import Flask, request, jsonify

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return "✅ Flask app is running on localhost!"

# @app.route('/webhook', methods=['POST'])
# def webhook():
#     req = request.get_json(force=True)

#     # Debug print to see what Dialogflow sends
#     print("👉 Received request from Dialogflow:")
#     print(req)

#     # Example: Extract the intent name
#     intent_name = req.get('queryResult', {}).get('intent', {}).get('displayName', 'Unknown')

#     # Send a simple text response back
#     response = {
#         'fulfillmentText': f"You triggered the intent: {intent_name}"
#     }

#     return jsonify(response)

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify
import torch
import pickle
import spacy
from nbs.bags_of_tricks import FastText

# === Load assets ===
MODEL_PATH = "intent_model.pt"
VOCAB_PATH = "vocab.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

# Load spaCy tokenizer
nlp = spacy.load("en_core_web_sm")

def tokenize(text):
    return [tok.text for tok in nlp.tokenizer(text)]

def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(" ".join(n_gram))
    return x

# Load vocab and label encoder
with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)
with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# Load model
model = FastText(vocab_size=len(vocab), embedding_dim=100, output_dim=len(label_encoder.classes_))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# === Flask setup ===
app = Flask(__name__)

def predict_intent(text):
    tokens = generate_bigrams(tokenize(text.lower()))
    indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
    tensor = torch.LongTensor(indices).unsqueeze(1)  # [seq_len, 1]
    with torch.no_grad():
        output = model(tensor)
        prediction = output.argmax(dim=1).item()
    return label_encoder.inverse_transform([prediction])[0]

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(force=True)
    user_input = req.get("queryResult", {}).get("queryText", "")
    print("📩 User input:", user_input)

    predicted_intent = predict_intent(user_input)
    print("🎯 Predicted Intent:", predicted_intent)

    response_text = generate_response(predicted_intent)
    return jsonify({'fulfillmentText': response_text})

def generate_response(intent):
    # Customize this dictionary with actual replies
    responses = {
        "greeting": "Hey! How can I assist you today?",
        "book_hotel": "Sure, I can help you find a hotel.",
        "weather": "Let me check the weather for you.",
        "goodbye": "Goodbye! Have a great day!",
        "fallback": "I'm not sure I understood that. Can you rephrase?"
    }
    return responses.get(intent, responses["fallback"])

if __name__ == '__main__':
    app.run(port=5000, debug=True)
