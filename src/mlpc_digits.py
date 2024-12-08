from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from collections import defaultdict

digits = load_digits()
X, y = digits.data, digits.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Multi-layer Perceptron classifier
mlp = MLPClassifier(random_state=42)

param_grid = {
    'hidden_layer_sizes': [(50, ), (100, 100), (300, 300)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'max_iter': [90, 100, 200]
}

grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)

grid_search.fit(X_train, y_train)

print("Best parameters: ", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)
best_mlp = grid_search.best_estimator_
y_predicted = best_mlp.predict(X_test)

confusion = confusion_matrix(y_test, y_predicted)
accuracy = accuracy_score(y_test, y_predicted)
precision = precision_score(y_test, y_predicted, average='macro')
recall = recall_score(y_test, y_predicted, average='macro')
f1 = f1_score(y_test, y_predicted, average='macro')
print(f"Confusion Matrix: \n{confusion}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Plot grid search results
results = grid_search.cv_results_
mean_test_scores = results['mean_test_score']
param_combinations = range(len(mean_test_scores))

# Prepare parameter combination labels for the legend
param_labels = [
    f"Index {i}:AF={params['activation']}, HL={params['hidden_layer_sizes']}, MI={params['max_iter']}"
    for i, params in enumerate(results['params'])
]

# Group mean test scores by activation function
grouped_scores = defaultdict(list)
grouped_indices = defaultdict(list)

for idx, params in enumerate(results['params']):
    activation = params['activation']
    grouped_scores[activation].append(mean_test_scores[idx])
    grouped_indices[activation].append(idx)

# Plot mean test scores for each activation function
plt.figure(figsize=(14, 7))
for activation, scores in grouped_scores.items():
    plt.plot(grouped_indices[activation], scores, marker='o', label=f"Activation: {activation}")

# Add a legend with parameter combination labels
for i, label in enumerate(param_labels):
    plt.plot([], [], ' ', label=label)  # Add empty plot for each parameter combination

# Add title, labels, and legend
plt.title("Grid Search Mean Test Scores with Parameter Combinations", fontsize=14)
plt.xlabel("Parameter Combination Index", fontsize=12)
plt.ylabel("Mean Test Score", fontsize=12)
plt.xticks(param_combinations, rotation=90, fontsize=10)
plt.grid()
plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))

# Show the plot
plt.tight_layout()
plt.show()
