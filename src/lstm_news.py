import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.src.utils import pad_sequences, to_categorical
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Embedding, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import stanza

stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma')


# Custom preprocessor to exclude tokens that are only digits
def preprocessor(text):
    text = text.lower()
    print(f'\nText before Regex:\n', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Normalize multiple spaces to a single space
    text = re.sub(r'\s+', ' ', text).strip()
    print(f'\nText after Regex ({len(text)}): ', text)
    # Strip leading and trailing spaces
    doc = nlp(text)
    # print(f'\nText from nlp: ', doc)
    text = ' '.join([word.lemma for sentence in doc.sentences for word in sentence.words])
    print(f'Text after Lemma ({len(text)}):', text)
    # # Remove special characters and punctuation
    # text = re.sub(r'[^\w\s]', '', text)  # Retain only words and spaces
    # # Remove tokens that are entirely numeric
    # text = re.sub(r'\b\d+\b', '', text)
    # Remove stop words
    tokens = text.split()
    # print('Text split: ', tokens)
    tokens = ' '.join([word for word in tokens if word not in ENGLISH_STOP_WORDS])
    print(f'Without stop words ({len(tokens)}):', tokens)
    return tokens


# print(f"ENGLISH_STOP_WORDS:\n{tokens}")

data = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))

# Extract the data (text) and target (categories)
X = data.data  # The text documents
y = data.target  # The target categories

print('Labels:')
for index, label in enumerate(data.target_names):
    print(f"{index}. {label}")

# Encode labels
label_encoder = LabelEncoder()
y_encoded_labels = label_encoder.fit_transform(y)

# Apply preprocessing to each document
print('Preprocessing started.')
X_preproc = [preprocessor(doc) for doc in X]
print('Preprocessing done.')

print(f"\nNumber of Words in BoW: {len(X_preproc)}")
print(f"Bag of Words (BoW) / Vocabulary Keys:\n{X_preproc[:100]}")

# Train-test split
X_train_val, X_test, y_train_val, y_test = train_test_split(X_preproc, y_encoded_labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

print(f"X_train:\n{X_train[:100]}")

max_vocab_size = 20000
max_sequence_length = 100

tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<UNK>")
tokenizer.fit_on_texts(X_train)

train_sequence = tokenizer.texts_to_sequences(X_train)
test_sequence = tokenizer.texts_to_sequences(X_test)
val_sequence = tokenizer.texts_to_sequences(X_val)

X_train = pad_sequences(train_sequence, max_sequence_length, padding='post')
X_test = pad_sequences(test_sequence, max_sequence_length, padding='post')
X_val = pad_sequences(val_sequence, max_sequence_length, padding='post')

# Number of classes
num_classes = np.max(y_encoded_labels) + 1

# One-hot encode the validation labels
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
y_val = to_categorical(y_val, num_classes)

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Define GRU Model
lstm_model = Sequential([
    Embedding(input_dim=max_vocab_size, output_dim=128),
    LSTM(128),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax'),
])

lstm_model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

lstm_history = lstm_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val),
                              verbose=1, callbacks=[early_stopping])

val_loss, val_accuracy = lstm_model.evaluate(X_val, y_val)
print(f"\nLSTM Validation Accuracy: {val_accuracy:.4f}")

predictions = lstm_model.predict(X_test)
predicted_class = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

cm = confusion_matrix(true_labels, predicted_class)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot()
plt.title("Classification Matrix")
plt.show()
