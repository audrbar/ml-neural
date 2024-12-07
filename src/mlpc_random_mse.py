import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * (X ** 2) - 3 * X + 5 + np.random.normal(scale=10, size=X.shape)

X_train = X[:70]
y_train = y[:70]
X_test = X[70:]
y_test = y[70:]

degrees = [1, 2, 5]

plt.figure(figsize=(15, 5))
for i, degree in enumerate(degrees):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    plt.subplot(1, 3, i + 1)
    plt.scatter(X, y, color="gray", label="Data")
    plt.plot(X, model.predict(X), label=f"Model (Degree {degree})", color="red")
    plt.title(f"Degree {degree}\nTrain MSE: {train_mse:.2f}, Test MSE: {test_mse:.2f}")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
