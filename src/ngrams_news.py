import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
import re


# Custom preprocessor to exclude tokens that are only digits
def preprocessor(text):
    # Remove tokens that are entirely numeric
    return re.sub(r'\b\d+\b', '', text)


data = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))

# Explore the dataset
print(f"\nNumber of Documents: {len(data.data)}")
print(f"Number of Categories: {len(data.target_names)}")
print(f"\nAll Categories: {np.unique(data.target_names)}")
print(f"\nAll Targets: {np.unique(data.target)}")
print(f"\nSecond Document: \n{data.data[1]}")
print(f"\nTarget (Category index) for the second document: {data.target[1]}")
print(f"Target (Category name) for the second document: {data.target_names[data.target[1]]}")

# Extract the data (text) and target (categories)
X = data.data  # The text documents
y = data.target  # The target categories

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Initial Vectorized Bag of Words (BoW)
vectorizer_initial = CountVectorizer()
X_train_vectorized_initial = vectorizer_initial.fit_transform(X)
print(f"\nNumber of Words in Initial BoW: {len(vectorizer_initial.get_feature_names_out())}")
print(f"\nInitial Bag of Words (BoW) / Vocabulary Keys:\n{vectorizer_initial.get_feature_names_out()[:100]}")

# Cleaned Vectorized Bag of Words (BoW)
vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=10000, preprocessor=preprocessor, stop_words='english')

X_train_vectorized = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()
X_val = vectorizer.transform(X_val).toarray()
print(f"\nNumber of Words in Cleaned BoW: {len(vectorizer.get_feature_names_out())}")
print(f"\nCleaned Bag of Words (BoW):\n{vectorizer.get_feature_names_out()}")
print(f"\nVocabulary Keys: {sorted(vectorizer.vocabulary_.keys())}")
print(f"\nX_train[0] Vectorized: \n{X_train_vectorized[0]}")

# Label Encoding
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)
y_val_encoded = encoder.transform(y_val)
print(f"\ny_train Encoded: {y_train_encoded}")

X_train_reshaped = X_train_vectorized.reshape((X_train_vectorized.shape[0], X_train_vectorized.shape[1], 1))
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
X_val_reshaped = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
print(f"\ny_train Reshaped: {X_train_reshaped}")

model = Sequential([
    SimpleRNN(64, input_shape=(X_train_reshaped.shape[1], 1), activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_reshaped, y_train_encoded, epochs=10, batch_size=16, validation_data=(X_val_reshaped, y_val_encoded))

test_loss, test_accuracy = model.evaluate(X_test_reshaped, y_test_encoded)
print(f"\nVal Loss: {test_loss:.4f}, Val Accuracy: {test_accuracy:.4f}")
