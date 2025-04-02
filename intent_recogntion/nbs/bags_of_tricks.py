import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import spacy
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
import matplotlib.pyplot as plt
import pickle

# Set seed for reproducibility
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Load spaCy model
spacy_en = spacy.load("en_core_web_sm")

def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(" ".join(n_gram))
    return x

# Custom Dataset Class
class IntentDataset(Dataset):
    def __init__(self, texts, labels, vocab, label_encoder):
        self.texts = texts
        self.labels = label_encoder.transform(labels)
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = generate_bigrams(tokenize(self.texts[idx].lower()))
        indices = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        return torch.tensor(indices), torch.tensor(self.labels[idx])

# Model Definition
class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        super(FastText, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        pooled = embedded.mean(dim=0)
        return self.fc(pooled)

# Data Preparation
data_path = "C:/Users/Jason/Documents/GitHub/intent-recognition/train.csv"
df = pd.read_csv(data_path)
texts, labels = df['text'].tolist(), df['label'].tolist()

# Label Encoding
label_encoder = LabelEncoder()
label_encoder.fit(labels)
num_classes = len(label_encoder.classes_)

# Vocabulary
tokenized_texts = [generate_bigrams(tokenize(text.lower())) for text in texts]
all_tokens = [token for sublist in tokenized_texts for token in sublist]
vocab_counter = Counter(all_tokens)
vocab = {token: idx+2 for idx, (token, _) in enumerate(vocab_counter.most_common(25000))}
vocab['<pad>'] = 0
vocab['<unk>'] = 1

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.15, random_state=SEED)

train_dataset = IntentDataset(train_texts, train_labels, vocab, label_encoder)
val_dataset = IntentDataset(val_texts, val_labels, vocab, label_encoder)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=lambda x: collate_fn(x, vocab['<pad>']))
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=lambda x: collate_fn(x, vocab['<pad>']))

def collate_fn(batch, pad_idx):
    texts, labels = zip(*batch)
    lengths = [len(x) for x in texts]
    max_len = max(lengths)
    padded = torch.full((len(texts), max_len), pad_idx, dtype=torch.long)
    for i, text in enumerate(texts):
        padded[i, :len(text)] = text
    return padded.T, torch.tensor(labels)

# Initialize Model
model = FastText(vocab_size=len(vocab), embedding_dim=100, output_dim=num_classes)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Training Loop
def train(model, iterator):
    model.train()
    total_loss = 0
    for batch in iterator:
        text, labels = batch
        optimizer.zero_grad()
        predictions = model(text)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(iterator)

def evaluate(model, iterator):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in iterator:
            text, labels = batch
            predictions = model(text)
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            correct += (predictions.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(iterator), correct / total

def get_predictions(model, iterator):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in iterator:
            text, labels = batch
            predictions = model(text)
            all_preds.extend(predictions.argmax(1).tolist())
            all_labels.extend(labels.tolist())
    return all_labels, all_preds

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(xticks_rotation=90, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

for epoch in range(10):
    train_loss = train(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

# Save vocab
with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

# Save label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Evaluation
print("\nGenerating Evaluation Report...")
y_true, y_pred = get_predictions(model, val_loader)
plot_confusion_matrix(y_true, y_pred)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

torch.save(model.state_dict(), "BoT.pt")

