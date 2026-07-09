import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../datasets/study_hours_marks.csv")

plt.scatter(data["Hours_Studied"], data["Marks"])
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.show()


def loss_function(m, b, points):

    total_error = 0

    for i in range(len(points)):
        x = points.iloc[i].Hours_Studied
        y = points.iloc[i].Marks
        total_error += (y - (m * x + b)) ** 2

    return total_error / float(len(points))


def gradient_descent(m_now, b_now, point, L):
    m_gradient = 0
    b_gradient = 0

    n = len(point)

    for i in range(n):
        if i % 50 == 0:
            print(f"Epoch: {i}, Loss: {loss_function(m_now, b_now, point)}")

        x = point.iloc[i].Hours_Studied
        y = point.iloc[i].Marks

        m_gradient += -(2 / n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2 / n) * (y - (m_now * x + b_now))

    m = m_now - L * m_gradient
    b = b_now - L * b_gradient

    return m, b


m = 0
b = 0
L = 0.0001
epochs = 1000

for i in range(epochs):
    m, b = gradient_descent(m, b, data, L)

print(f"Final values after {epochs} epochs: \nm = {m}, \nb = {b}")

x_val = data["Hours_Studied"]
y_val = m * x_val + b

plt.scatter(data["Hours_Studied"], data["Marks"])
plt.plot(x_val, y_val, color="red")
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.show()
