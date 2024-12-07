import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load the dataset
digits = load_digits()
X, y = digits.data, digits.target

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Define the parameter grid
param_grid = {
    'solver': ['adam', 'sgd'],
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': [0.001, 0.01],
    'alpha': [0.0001, 0.001],
}
mlp = MLPClassifier(hidden_layer_sizes=(300, 300), activation='relu', max_iter=200, random_state=42)
# Perform grid search
grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, refit=True, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Best model and parameters
print("Best parameters: ", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

best_mlp = grid_search.best_estimator_
y_predicted = best_mlp.predict(X_test)

confusion = confusion_matrix(y_test, y_predicted)
accuracy = accuracy_score(y_test, y_predicted)
loss = log_loss(y_test, best_mlp.predict_proba(X_test))

print(f"Confusion Matrix: \n{confusion}")
print(f"Accuracy: {accuracy}")
print(f"Log Loss: {loss}")

# Compute Cross-Entropy Loss (Log Loss) for each parameter set
log_losses = []
for idx, params in enumerate(grid_search.cv_results_['params']):
    # Use the best model for each parameter set
    model = MLPClassifier(**params, hidden_layer_sizes=(300, 300), activation='relu', max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)
    loss = log_loss(y_test, y_pred_proba)
    log_losses.append(loss)

# Assign colors to each solver for differentiation
solvers = list(set([p['solver'] for p in grid_search.cv_results_['params']]))
colors = cm.rainbow(np.linspace(0, 1, len(solvers)))

# Visualize Cross-Entropy Loss with color coding
plt.figure(figsize=(17, 7))
for idx, params in enumerate(grid_search.cv_results_['params']):
    solver = params['solver']
    color = colors[solvers.index(solver)]  # Assign color based on solver
    plt.scatter(idx, log_losses[idx], color=color, label=f"{solver}" if idx == 0 else "")

plt.title("Cross-Entropy Loss for Each Parameter Set in Grid Search")
plt.xlabel("Parameter Combination Index")
plt.ylabel("Cross-Entropy Loss")
plt.xticks(range(len(log_losses)), [str(p) for p in grid_search.cv_results_['params']], rotation=90)
plt.legend(loc="upper right", title="Solver", fontsize="small", frameon=True)
plt.grid()
plt.tight_layout()
plt.show()
