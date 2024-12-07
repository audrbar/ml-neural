import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, make_scorer

data = load_breast_cancer()
# data = load_iris()
X = data.data
y = (data.target != 0).astype(int)


models = [
    ("Underfitting", MLPClassifier(hidden_layer_sizes=(5, 2), max_iter=150, random_state=42)),
    ("Right Fit", MLPClassifier(hidden_layer_sizes=(20, 20), max_iter=400, random_state=42)),
    ("Overfitting", MLPClassifier(hidden_layer_sizes=(30, 30), max_iter=800, random_state=42))
]


def plot_learning_curve(model_, title_, ax_):
    pipeline = make_pipeline(StandardScaler(), model_)
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X, y, cv=5, scoring=make_scorer(accuracy_score), train_sizes=np.linspace(0.1, 0.8, 10)
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)

    ax_.plot(train_sizes, train_scores_mean, label="Training Accuracy", marker="o", color="blue")
    ax_.plot(train_sizes, val_scores_mean, label="Validation Accuracy", marker="o", color="orange")
    ax_.set_title(f"Learning Curve\n{title_}")
    ax_.set_xlabel("Training Set Size")
    ax_.set_ylabel("Accuracy")
    ax_.legend()
    ax_.grid(True)


fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for (title, model), ax in zip(models, axes):
    plot_learning_curve(model, title, ax)

plt.tight_layout()
plt.show()
