import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("../datasets/study_hours_marks.csv")

X = data[["Hours_Studied"]]
y = data["Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=45
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Slope: ", model.coef_[0])
print("Intercept: ", model.intercept_)
print(f"Equation: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")

print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression using Scikit-learn")
plt.legend()
plt.show()
