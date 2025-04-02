import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
data_path = "C:/Users/Jason/Documents/GitHub/intent-recognition/train.csv"
df = pd.read_csv(data_path) # Change if using valid/test separately
texts = df['text'].tolist()
labels = df['label'].tolist()

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased").to(device)

# Encode sentences with BERT
def get_cls_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # [CLS] token

print("Encoding sentences with BERT (this may take a while)...")
X = np.array([get_cls_embedding(text) for text in texts])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost classifier
clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot confusion matrix and save as JPG
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("transformer_xgboost_confusion_matrix.jpg")
plt.close()

# Save XGBoost model and label encoder
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(clf, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("\nModel, encoder, and confusion matrix image saved!")
