import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("../datasets/study_hours_marks.csv")

X = data["Hours_Studied"].values.reshape(-1, 1)
y = data["Marks"].values.reshape(-1, 1)

X_b = np.c_[
    np.ones((X.shape[0], 1)), X
]  # add bias term (intercept) to the input features

theta = np.zeros((2, 1))  # initialize parameters (weights) to zero

learning_rate = 0.0001
epochs = 1000
m = len(y)

for _ in range(epochs):
    predictions = X_b @ theta
    gradient = -(2 / m) * X_b.T @ (predictions - y)
    theta += learning_rate * gradient

print("Bias:", theta[0, 0])
print("Weight:", theta[1, 0])

# Predictions
y_pred = X_b @ theta
print(y_pred.ravel())

plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, y_pred, color="red", linewidth=2, label="Regression Line")

plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Linear Regression using Gradient Descent")
plt.legend()

plt.show()
