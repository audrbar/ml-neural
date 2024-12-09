import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix


data = load_iris()
X = data.data
y = data.target

encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)
X_test = scaler.transform(X_test)

# Paversti duom i tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Sukuriam duomenu rinkinius
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


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


model = SimpleModel(input_size=X_train_tensor.shape[1], output_size=y_train_tensor.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Apmokame modeli
for epoch in range(10):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, torch.argmax(y_batch, axis=1))
        # Skaiciuojame gradientus
        loss.backward()
        # Atnaujiname svorius
        optimizer.step()

    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0
    for X_val, y_val in val_loader:
        val_output = model(X_val)
        val_loss += criterion(val_output, torch.argmax(y_val, axis=1)).item()
        _, val_preds = torch.max(val_output, 1)
        val_correct += (val_preds == torch.argmax(y_val, axis=1)).sum().item()
        val_total += y_val.size(0)
    val_acc = val_correct/val_total
    val_loss /= len(val_loader)

    print(f"Epoch {epoch + 1}, train_loss: {loss.item():.4f}, val_acc: {val_acc:.4f}, val_loss: {val_loss:.4f}")


model.eval()
y_pred_probs = model(X_test_tensor)
y_pred = torch.argmax(y_pred_probs, axis=1).numpy()
y_test_labels = torch.argmax(y_test_tensor, axis=1).numpy()

acc = accuracy_score(y_test_labels, y_pred)
print('Accuracy: ', acc)