import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
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


# Define a function to categorize experience
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


# Function to dynamically create optimizer with learning rate schedule
def get_optimizer(optimizer_name, initial_lr=0.01, decay_steps=100, decay_rate=0.96):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate
    )
    if optimizer_name == 'adam':
        return Adam(learning_rate=lr_schedule)
    elif optimizer_name == 'sgd':
        return SGD(learning_rate=lr_schedule)
    elif optimizer_name == 'rmsprop':
        return RMSprop(learning_rate=lr_schedule)
    elif optimizer_name == 'adagrad':
        return Adagrad(learning_rate=lr_schedule)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


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

# Data Preparation
X = df.drop(columns=['Segmentation']).values  # Features
y = df['Segmentation'].values  # True labels for evaluation

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)


# Function to build, compile, and train the model
def train_model(optimizer_name, initial_lr=0.01, decay_steps=10, decay_rate=0.96, batch_size=16, epochs=100):
    # Model Definition
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(len(np.unique(y)), activation='softmax')
    ])

    optimiser = get_optimizer(optimizer_name, initial_lr, decay_steps, decay_rate)

    # Compile Model
    model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Model Training
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[early_stopping]
    )

    # Predictions and Evaluation
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy with initial_lr={initial_lr}, decay_steps={decay_steps}, decay_rate={decay_rate}: {acc:.4f}')

    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:\n', cm)

    steps_per_epoch = X_train.shape[0] / 16
    print('Steps per Epoch: ', steps_per_epoch)

    # Store metrics
    metrics = {
        'optimizer': optimizer_name,
        'initial_lr': initial_lr,
        'decay_steps': decay_steps,
        'decay_rate': decay_rate,
        'accuracy': acc,
        'validation_loss': history.history['val_loss'][-1]
    }

    # Visualization
    plt.figure(figsize=(15, 7))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss during Training ({optimizer_name})')
    plt.legend()
    plt.grid()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy during Training ({optimizer_name})')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

    return metrics


# Initialize an empty list, train models with different configurations and store metrics
all_metrics = [
    train_model(optimizer_name='adam', initial_lr=0.01, decay_steps=100, decay_rate=0.96),
    train_model(optimizer_name='adam', initial_lr=0.001, decay_steps=100, decay_rate=0.96),
    train_model(optimizer_name='sgd', initial_lr=0.01, decay_steps=50, decay_rate=0.9),
    train_model(optimizer_name='sgd', initial_lr=0.001, decay_steps=50, decay_rate=0.9),
    train_model(optimizer_name='rmsprop', initial_lr=0.01, decay_steps=200, decay_rate=0.95),
    train_model(optimizer_name='rmsprop', initial_lr=0.001, decay_steps=200, decay_rate=0.96),
    train_model(optimizer_name='adagrad', initial_lr=0.01, decay_steps=200, decay_rate=0.95),
    train_model(optimizer_name='adagrad', initial_lr=0.001, decay_steps=200, decay_rate=0.96),
]

# Convert metrics to a DataFrame for tabular display and analysis
metrics_df = pd.DataFrame(all_metrics)

# Display the table
print("\nModel Configurations and Results:")
print(metrics_df)

# Find the best configuration based on validation loss
best_model = min(all_metrics, key=lambda x: x['validation_loss'])

print("\nBest Model Configuration:")
for key, value in best_model.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

# Plot performance metrics
plt.figure(figsize=(15, 7))
plt.bar(metrics_df['optimizer'], metrics_df['accuracy'], label='Accuracy', alpha=0.6)
plt.xlabel('Optimizer')
plt.ylabel('Accuracy')
plt.title('Optimizer Performance Comparison')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.show()
