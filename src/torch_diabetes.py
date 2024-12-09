import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


pd.options.display.max_columns = None
initial_df = pd.read_csv('/Users/audrius/Documents/VCSPython/ml-neural-net/data/diabetes.csv')

data = initial_df.dropna()

X = data.drop(columns=['Outcome']).values
y = data['Outcome'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)
X_test = scaler.fit_transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# Classifier Neural Network
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fn = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fn(x)


# Define optimizers and learning rates to cycle through
optimizers = {
    'Adadelta': optim.Adadelta,
    'Adafactor': optim.Adafactor,
    'Adagrad': optim.Adagrad,
    'Adam': optim.Adam,
    'AdamW': optim.AdamW,
    'ASGD': optim.ASGD,
    'NAdam': optim.NAdam,
    'RAdam': optim.RAdam,
    'RMSprop': optim.RMSprop,
    'Rprop': optim.Rprop,
    'SGD': optim.SGD
}
learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]

# Store results for plotting
results = []
# Training loop for each optimizer and learning rate
for opt_name, opt_class in optimizers.items():
    for lr in learning_rates:
        print(f"Training with {opt_name}, Learning Rate: {lr}")
        # Initialize model, criterion, and optimizer
        model = SimpleModel(input_size=X_train_tensor.shape[1], output_size=y_train_tensor.shape[1])
        criterion = nn.CrossEntropyLoss()
        optimizer = opt_class(model.parameters(), lr=lr)

        # Training
        for epoch in range(10):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, torch.argmax(y_batch, axis=1))
                loss.backward()
                optimizer.step()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        for X_val, y_val in val_loader:
            val_output = model(X_val)
            _, val_preds = torch.max(val_output, 1)
            val_correct += (val_preds == torch.argmax(y_val, axis=1)).sum().item()
            val_total += y_val.size(0)
        val_acc = val_correct / val_total

        # Test accuracy
        y_pred_probs = model(X_test_tensor)
        y_pred = torch.argmax(y_pred_probs, axis=1).numpy()
        y_test_labels = torch.argmax(y_test_tensor, axis=1).numpy()
        test_acc = accuracy_score(y_test_labels, y_pred)

        print(f"{opt_name}, LR: {lr}, Val Accuracy: {val_acc:.4f}, Test Accuracy: {test_acc:.4f}")

        # Store the results
        results.append({'Optimizer': opt_name, 'Learning Rate': lr, 'Validation Accuracy': val_acc, 'Test Accuracy': test_acc})

# Convert results to a DataFrame for visualization
results_df = pd.DataFrame(results)

# Find the best result based on Test Accuracy
best_result = results_df.loc[results_df['Test Accuracy'].idxmax()]

# Print the best configuration and its score
print(f"\nBest Configuration: Optimizer: {best_result['Optimizer']}, Learning Rate: {best_result['Learning Rate']}, "
      f"")
print(f"Validation Accuracy: {best_result['Validation Accuracy']:.4f}")
print(f"Test Accuracy: {best_result['Test Accuracy']:.4f}")

# Plot results
plt.figure(figsize=(14, 7))
for opt_name in optimizers.keys():
    subset = results_df[results_df['Optimizer'] == opt_name]
    plt.plot(subset['Learning Rate'], subset['Test Accuracy'], marker='o', label=f"{opt_name}")

plt.xscale('log')  # Use a logarithmic scale for learning rates
plt.xlabel("Learning Rate (Log Scale)")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy by Optimizer and Learning Rate")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# torch.save(model.state_dict(), "data/model.pth")
# loaded_model = SimpleModel(input_size=X_train_tensor.shape[1], output_size=y_train_tensor.shape[1])
# loaded_model.load_state_dict(torch.load("data/model.pth"))
# predictions = loaded_model(X_test_tensor)
# predicted_class = torch.argmax(predictions, axis=1)
# print('Predicted: ', predicted_class)
#
# with open('model_state.pt', 'wb') as f:
#     save(model.state_dict(), f)

# with open('model_state.pt', 'rb') as f:
#     model.load_state_dict(load(f))
#
# img = Image.open('img/img_3.jpg')
# img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')
#
# print(torch.argmax(clf(img_tensor)))
