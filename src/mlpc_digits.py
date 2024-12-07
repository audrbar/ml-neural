from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

digits = load_digits()
X, y = digits.data, digits.target
print(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

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

plt.figure(figsize=(12, 6))
plt.plot(param_combinations, mean_test_scores, marker='o')
plt.title("Grid Search Mean Test Scores")
plt.xlabel("Parameter Combination Index")
plt.ylabel("Mean Test Score")
plt.xticks(param_combinations, rotation=90)
plt.grid()
plt.show()