import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix


# Define a function to categorize age
def categorize_age(age):
    if age < 21:
        return 0  # Youth
    elif age < 35:
        return 1  # Young Adult
    elif age < 50:
        return 2  # Adult
    elif age < 65:
        return 3  # Mature Adult
    else:
        return 4  # Senior


def categorize_experience(experience):
    if experience < 1:
        return 0  # very low
    elif experience < 3:
        return 1  # low
    elif experience < 6:
        return 2  # moderate
    elif experience < 9:
        return 3  # high
    elif experience < 12:
        return 4  # very high
    else:
        return 5  # exceptional


# Load Data
pd.options.display.max_columns = None
initial_df = pd.read_csv('/Users/audrius/Documents/VCSPython/ml-clustering-automobile/data/train-set.csv')

# Clean and Prepare Data
df = initial_df.drop(columns=['CustomerID']).dropna()

# Apply the functions or mapping to the column to categorize data instead using LabelEncoder
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Married'] = df['Married'].map({'No': 0, 'Yes': 1})
df['Graduated'] = df['Graduated'].map({'No': 0, 'Yes': 1})
df['Profession'] = df['Profession'].map({'Healthcare': 0, 'Engineer': 1, 'Lawyer': 2, 'Entertainment': 3,
                                         'Artist': 4, 'Executive': 5, 'Doctor': 6, 'Homemaker': 7, 'Marketing': 8
                                         }).fillna(-1)
df['SpendingScore'] = df['SpendingScore'].map({'Low': 0, 'Average': 1, 'High': 2})
df['Category'] = df['Category'].map({'Category 1': 0, 'Category 2': 1, 'Category 3': 2, 'Category 4': 3,
                                     'Category 5': 4, 'Category 6': 5, 'Category 7': 6})
df['Segmentation'] = df['Segmentation'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3})
X_hist = df.drop(columns=['Segmentation'])
df['Age'] = df['Age'].apply(categorize_age)
df['WorkExperience'] = df['WorkExperience'].apply(categorize_experience)

# Prepare Data for Clustering
X = df.drop(columns=['Segmentation']).values  # Features
y = df['Segmentation'].values  # True labels for evaluation

print(X)
print(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# [0, 0, 1], [0, 1, 0], [1, 0, 0]
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(y_train.shape[1], activation='sigmoid')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=40, batch_size=32, verbose=1)

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test_labels, y_pred)
print(f'Accuracy:, {acc:.4f}')

cm = confusion_matrix(y_test_labels, y_pred)
print('Confusion Matrix:\n', cm)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss during training')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy during training')
plt.legend()
plt.show()
