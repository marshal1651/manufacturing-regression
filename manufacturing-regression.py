import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load Dataset
df = pd.read_csv('/storage/emulated/0/ml_projects/gh/manufacturing.csv')

# Optional: Basic data inspection (commented)

#print(df.head())  
#print(df.columns)
#print(df.shape)
#for i in df.columns:
#	print(df[i]) 
#print(df.info())
#print(df.describe())
#print(df.isnull().sum())
#print(df.duplicated().sum())

# ------------------------ Pair Plot ------------------------ #
sns.pairplot(df)
plt.suptitle("Pairplot of Scaled Features vs Quality Rating")
plt.tight_layout()
plt.show()

# Drop low-correlation or irrelevant feature
df = df.drop(['Pressure (kPa)'], axis=1)

# Feature and Target Separation
X = df.iloc[:, :-1]
y = df['Quality Rating']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------ Linear Regression ------------------------ #
lr = LinearRegression()
lr.fit(x_train, y_train)

print("\n--- Linear Regression Results ---")
print(f"Coefficients       : {lr.coef_}")
print(f"Intercept          : {lr.intercept_}")
print(f"R² Score (Train)   : {lr.score(x_train, y_train)*100:.2f}%")
print(f"R² Score (Test)    : {lr.score(x_test, y_test)*100:.2f}%")

# Predict and calculate MSE
y_pred_linear = lr.predict(x_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
print(f"MSE (Test)         : {mse_linear:.2f}")

# ------------------------ Polynomial Regression ------------------------ #
poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

lr_poly = LinearRegression()
lr_poly.fit(x_train_poly, y_train)

print("\n--- Polynomial Regression Results ---")
print(f"Coefficients       : {lr_poly.coef_}")
print(f"Intercept          : {lr_poly.intercept_}")
print(f"R² Score (Train)   : {lr_poly.score(x_train_poly, y_train)*100:.2f}%")
print(f"R² Score (Test)    : {lr_poly.score(x_test_poly, y_test)*100:.2f}%")

# Predict and calculate MSE
y_pred_poly = lr_poly.predict(x_test_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
print(f"MSE (Test)         : {mse_poly:.2f}")

# ------------------------ Visualization ------------------------ #
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred_poly, alpha=0.6, color='green')
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.title("Polynomial Regression: Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.savefig("/storage/emulated/0/ml_projects/polynomial_actual_vs_predicted.png")  # Optional
plt.show()