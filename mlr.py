
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv("Housing.csv") 


print(df.head())
print(df.info())


df = pd.get_dummies(df, drop_first=True)


print(df.isnull().sum())


X_simple = df[['area']]
y = df['price']


X_train, X_test, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)

# Train the model
model_simple = LinearRegression()
model_simple.fit(X_train, y_train)

# Predict and evaluate
y_pred = model_simple.predict(X_test)
print("Simple Linear Regression Evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Plotting
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Simple Linear Regression: Price vs Area")
plt.legend()
plt.show()

# ----------------- MULTIPLE LINEAR REGRESSION -----------------
# Select features
X_multi = df.drop('price', axis=1)
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_multi, y, test_size=0.2, random_state=42)

# Train model
model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

# Predict and evaluate
y_pred = model_multi.predict(X_test)
print("\nMultiple Linear Regression Evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Coefficients
coeff_df = pd.DataFrame(model_multi.coef_, X_multi.columns, columns=['Coefficient'])
print("\nFeature Coefficients:\n", coeff_df)
