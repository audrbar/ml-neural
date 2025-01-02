import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import nltk
import re
import spacy
from nltk.corpus import stopwords
from sklearn.utils.class_weight import compute_class_weight


nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')
newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

texts = newsgroups_data.data
labels = newsgroups_data.target

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    doc = nlp(text)
    text = ' '.join([token.lemma_ for token in doc])

    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


X_train_clean = [preprocess_text(text) for text in X_train]
X_test_clean = [preprocess_text(text) for text in X_test]

max_vocab_size = 10000
max_sequence_length = 200

tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_clean)

X_train_seq = tokenizer.texts_to_sequences(X_train_clean)
X_test_seq = tokenizer.texts_to_sequences(X_test_clean)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length, padding='post', truncating='post')

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

embedding_dim = 128
num_classes = 20

model = Sequential([
    Embedding(input_dim=max_vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
    GRU(128, return_sequences=False),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

batch_size = 32
epochs = 20

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

history = model.fit(X_train_pad, y_train, validation_data=(X_test_pad, y_test), epochs=epochs, batch_size=batch_size,
                    class_weights=class_weight_dict)

test_loss, test_accuracy = model.evaluate(X_test_pad, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy:.2f}")

sample_texts = ["NASA launched a new satellite.", "The hockey match was thrilling."]
sample_cleaned = [preprocess_text(text) for text in sample_texts]
sample_seq = tokenizer.texts_to_sequences(sample_cleaned)
sample_pad = pad_sequences(sample_seq, maxlen=max_sequence_length, padding='post', truncating='post')

predictions = model.predict(sample_pad)
predicted_classes = [label_encoder.inverse_transform([np.argmax(pred)])[0] for pred in predictions]

for text, category in zip(sample_texts, predicted_classes):
    print(f"Text: '{text}' --> Predicted Category: {category}")
