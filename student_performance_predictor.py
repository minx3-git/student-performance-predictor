print("RUNNING FILE NOW")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv("student_data.csv")

X = data[["hours_studied", "sleep_hours", "attendance"]]
y = data["score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print("Mean Absolute Error:", round(mae, 2))
print("R² Score:", round(r2, 2))

print("\n--- Predict New Student Score ---")
hours = float(input("Hours studied per day: "))
sleep = float(input("Sleep hours per day: "))
attendance = float(input("Attendance percentage: "))

new_score = model.predict([[hours, sleep, attendance]])
print("Predicted Exam Score:", round(new_score[0], 2))

# Convert to numpy arrays for consistent plotting
y_test_arr = np.array(y_test)
y_pred_arr = np.array(y_pred)

plt.scatter(y_test_arr, y_pred_arr)

# identity line y = x
mn = min(y_test_arr.min(), y_pred_arr.min())
mx = max(y_test_arr.max(), y_pred_arr.max())
plt.plot([mn, mx], [mn, mx], 'k--', label='y = x')

# model line (connect predicted points in order of true values)
order = np.argsort(y_test_arr)
plt.plot(y_test_arr[order], y_pred_arr[order], 'r-', label='model')

plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("Actual vs Predicted Student Scores")
plt.legend()
plt.show()
