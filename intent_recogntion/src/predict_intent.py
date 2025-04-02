import torch
import pickle
import spacy
from nbs.bags_of_tricks import FastText

# Load model and related assets
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

# Initialize model
model = FastText(vocab_size=len(vocab), embedding_dim=100, output_dim=len(label_encoder.classes_))
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

def predict_intent(text):
    tokens = generate_bigrams(tokenize(text.lower()))
    indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
    tensor = torch.LongTensor(indices).unsqueeze(1)  # [seq_len, 1]
    with torch.no_grad():
        output = model(tensor)
        prediction = output.argmax(dim=1).item()
    return label_encoder.inverse_transform([prediction])[0]

# Example usage
if __name__ == "__main__":
    example = "Find me a hotel in London"
    prediction = predict_intent(example)
    print(f"Predicted Intent: {prediction}")
