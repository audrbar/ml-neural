import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


data = fetch_20newsgroups()

x = data.data
y = data.target

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, encoded_labels, test_size=0.2, random_state=42)

max_vocab_size = 10000
max_sequence_length = 100

tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<UNK>")
tokenizer.fit_on_texts(x_train)

train_seq = tokenizer.texts_to_sequences(x_train)
test_seq = tokenizer.texts_to_sequences(x_test)

x_train = pad_sequences(train_seq, maxlen=max_sequence_length, padding='post', truncating='post')
x_test = pad_sequences(test_seq, maxlen=max_sequence_length, padding='post', truncating='post')

num_classes = np.max(encoded_labels) + 1
y_test = to_categorical(y_test, num_classes)
y_train = to_categorical(y_train, num_classes)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_vocab_size, output_dim=128, input_length=max_sequence_length),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax'),
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 32
epoch = 5

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=epoch, verbose=1)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("LSTM Test Accuracy: ", test_accuracy)

predictions = model.predict(x_test)
predicted_class = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

cm = confusion_matrix(true_labels, predicted_class)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot()
plt.title("Klasifikavimo matrica")
plt.show()
