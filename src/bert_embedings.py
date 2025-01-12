from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

newsgroups_data = fetch_20newsgroups(
    subset='all', remove=('headers', 'footers', 'quotes'), categories=['sci.space', 'comp.graphics']
)
X, y = newsgroups_data.data, newsgroups_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def extract_embedding(texts):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', max_length=32, padding='max_length')
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.append(cls_embedding.flatten())
    return np.array(embeddings)


def extract_embeddings(texts, batch_size=16):
    embeddings = []
    model.eval()  # Set model to evaluation mode
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        print(f"Bach Text: {batch_texts}")
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            max_length=128,
            truncation=True,
            padding='max_length'
        )
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.extend(cls_embeddings)
    return np.array(embeddings)


X_train_embeddings = extract_embeddings(X_train)
X_test_embeddings = extract_embeddings(X_test)

clf = LogisticRegression(max_iter=100)
clf.fit(X_train_embeddings, y_train)

y_pred = clf.predict(X_test_embeddings)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")
