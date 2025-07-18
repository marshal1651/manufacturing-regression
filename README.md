# Manufacturing Quality Rating Prediction

This project aims to predict the **Quality Rating** of a manufacturing process using regression models based on physical properties like temperature, humidity, torque, etc.

## 📌 Objective

To build a machine learning model that accurately predicts the quality rating of a manufacturing item using both **Linear Regression** and **Polynomial Regression**.

---

## 🗃️ Dataset

- Source: Local CSV file (`manufacturing.csv`)
- Target variable: `Quality Rating`
- Features include:
  - Temperature (°C)
  - Humidity (%)
  - Torque (Nm)
  - Load (Kg)
  - Vibration (Hz)
  - Speed (rpm)
  - [Pressure (kPa) was dropped due to low correlation]

---

## 📊 Exploratory Data Analysis (EDA)

- Used **Seaborn Pairplot** to visualize feature relationships.
- Identified features with low correlation and removed them to improve model performance.

---

## 🧪 Preprocessing

- Applied **Standard Scaling** to features for normalization.
- Performed **train-test split** (80/20) for evaluation.

---

## 🔍 Models Used

### 1. Linear Regression
- Trained on scaled features
- **Train R² Score**: ~95%
- **Test R² Score**: ~93%
- **MSE**: Low

### 2. Polynomial Regression (Degree = 2)
- Captured non-linear relationships
- **Train R² Score**: ~99%
- **Test R² Score**: ~95%
- **MSE**: Very low

---

## 📈 Visualization

- Scatter plot of **Actual vs Predicted** values for Polynomial Regression.
- Helps in evaluating model prediction quality visually.

---

## 🧰 Libraries Used

- pandas
- matplotlib
- seaborn
- sklearn (StandardScaler, LinearRegression, PolynomialFeatures, train_test_split, mean_squared_error)

---

## ✅ Conclusion

- **Polynomial Regression** outperformed simple Linear Regression on this dataset.
- The project demonstrates solid understanding of feature scaling, regression modeling, evaluation metrics, and visualization.

---