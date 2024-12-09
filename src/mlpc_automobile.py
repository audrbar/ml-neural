import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from collections import defaultdict


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

# # Display dataset information for exploration
# print("\nInitial Dataset Info:")
# print(initial_df.info())

# # Display unique values for each column
# print("\nInitial Dataset Unique Values for Each Column:")
# for col in initial_df.columns:
#     print(f"{col}: {initial_df[col].unique()}")

# Clean and Prepare Data
df = initial_df.drop(columns=['CustomerID']).dropna()  # Or: df = df.fillna(df.mean())

# Calculate target class balance and add percentage column
target_balance = df['Segmentation'].value_counts().reset_index()
target_balance.columns = ['Segmentation', 'Count']  # Rename columns for clarity
target_balance['Percentage'] = (target_balance['Count'] / target_balance['Count'].sum()) * 100
# print("\nTarget Class Balance:")
# for index, row in target_balance.iterrows():
#     print(f"{row['Segmentation']} - {row['Count']}, {row['Percentage']:.1f}%")

# Apply the functions or mapping to the column to categorize data instead using LabelEncoder
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Married'] = df['Married'].map({'No': 0, 'Yes': 1})
df['Graduated'] = df['Graduated'].map({'No': 0, 'Yes': 1})
df['Profession'] = df['Profession'].map({'Healthcare': 0, 'Engineer': 1, 'Lawyer': 2, 'Entertainment': 3,
                                         'Artist': 4, 'Executive': 5, 'Doctor': 6, 'Homemaker': 7, 'Marketing': 8})
df['SpendingScore'] = df['SpendingScore'].map({'Low': 0, 'Average': 1, 'High': 2})
df['Category'] = df['Category'].map({'Category 1': 0, 'Category 2': 1, 'Category 3': 2, 'Category 4': 3,
                                     'Category 5': 4, 'Category 6': 5, 'Category 7': 6})
df['Segmentation'] = df['Segmentation'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3})
X_hist = df.drop(columns=['Segmentation'])
df['Age'] = df['Age'].apply(categorize_age)
df['WorkExperience'] = df['WorkExperience'].apply(categorize_experience)

# Final inspection of the preprocessed dataset
print("\nCleaned and Preprocessed Dataset Info:")
print(df.info())
print("\nCleaned Dataset Unique Values for Each Column:")
for col in df.columns:
    print(f"{col}: {df[col].unique()}")

# Prepare Data for Clustering
X_pre = df.drop(columns=['Segmentation']).values  # Features
y_true = df['Segmentation'].values  # True labels for evaluation

# Apply scaling (StandardScaler) on the X_pre
scaler = StandardScaler()
X = scaler.fit_transform(X_pre)

encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y_true.reshape(-1, 1))

# # Final preprocessed data for analysis
# print("\nFinal preprocessed X:")
# print(X)
# print("\nFinal preprocessed y_true:")
# print(y_true)

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.1, random_state=42)

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

# confusion = confusion_matrix(y_test, y_predicted)
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
    f"{i}:AF={params['activation']}, HL={params['hidden_layer_sizes']}, MI={params['max_iter']}"
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
# plt.plot(param_combinations, mean_test_scores, marker='o', label="Mean Test Scores")
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
