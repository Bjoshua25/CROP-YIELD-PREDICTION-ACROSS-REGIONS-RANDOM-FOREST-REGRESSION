# **CROP YIELD PREDICTION ACROSS REGIONS | RANDOM FOREST**  
*Agricultural Data Analysis & Machine Learning*  

## **INTRODUCTION**  
Agricultural productivity varies across regions due to **climatic, soil, and farm management factors**. This project applies **Random Forest Regression** to predict **crop yield across different regions** using features like **temperature, rainfall, soil type, irrigation, and pesticide usage**.  

By leveraging **ensemble learning**, the model provides **accurate and reliable yield predictions** to assist farmers and policymakers.  

---

## **PROBLEM STATEMENT**  
Crop yield prediction is essential for:  
- **Farmers** – Optimizing agricultural practices based on environmental conditions.  
- **Government Agencies** – Planning food production and resource allocation.  
- **Sustainability Efforts** – Ensuring food security in different regions.  

This project aims to:  
- **Analyze regional agricultural data** to uncover yield trends.  
- **Train a Random Forest model** to predict crop yield based on key factors.  
- **Evaluate model accuracy** using standard regression metrics.  

---

## **SKILL DEMONSTRATION**  
- **Data Preprocessing & Feature Engineering**  
- **Exploratory Data Analysis (EDA) & Visualization**  
- **Random Forest Regression Modeling**  
- **Hyperparameter Tuning & Model Optimization**  
- **Model Evaluation (MSE, RMSE, R² Score)**  

---

## **DATA SOURCING**  
The dataset is sourced from [Explore-AI Public Data](https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Data/Python/Crop_yield.csv) and includes:  

### **1. Climatic & Soil Factors**  
- **Temperature (°C)**  
- **Rainfall (mm)**  
- **Soil Type** (Loamy, Clayey, Sandy)  

### **2. Farm Management Practices**  
- **Fertilizer Usage (kg/ha)**  
- **Pesticide Usage (kg/ha)**  
- **Irrigation (Binary: 0 = No, 1 = Yes)**  

### **3. Crop Yield & Region Information**  
- **Region** (North, South, East, West)  
- **Crop Variety** (Variety A, B, C)  
- **Yield (tons per hectare)**  

---

## **EXPLORATORY DATA ANALYSIS (EDA)**  
EDA was performed to **understand the impact of different factors on crop yield**.  

### **1. Data Overview**  
- **Checked dataset structure** using `.info()` and `.describe()`.  
- **Identified missing values** and handled inconsistencies.  

### **2. Crop Yield Distribution**  
- **Histogram to analyze yield variability**  
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,6))
sns.histplot(df['Yield'], bins=30, kde=True)
plt.xlabel('Crop Yield (tons/ha)')
plt.ylabel('Frequency')
plt.title('Distribution of Crop Yield')
plt.show()
```
- **Key Insight:** Crop yields vary significantly across regions.  

### **3. Correlation Analysis**  
- **Heatmap to analyze feature relationships:**  
```python
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Agricultural Features')
plt.show()
```
- **Key Finding:** Fertilizer and irrigation positively impact crop yield.  

---

## **RANDOM FOREST MODEL**  
A **Random Forest Regression Model** was trained to predict crop yield.  

### **1. Model Implementation**  
- **Independent Variables (`X`)**: Climatic, soil, and farm management factors.  
- **Dependent Variable (`y`)**: Crop Yield.  
- **Model Used**: `sklearn.ensemble.RandomForestRegressor`  

```python
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```

### **2. Model Evaluation**  
- **Mean Squared Error (MSE)**  
- **Root Mean Squared Error (RMSE)**  
- **R² Score (Explained Variance)**  

```python
from sklearn.metrics import mean_squared_error, r2_score

y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

---

## **MODEL INTERPRETATION & VISUALIZATION**  
### **1. Feature Importance**  
Identifying the most **impactful factors on crop yield predictions**:  
```python
import pandas as pd

feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title('Feature Importance in Random Forest Model')
plt.show()
```
- **Key Finding:** Rainfall, soil type, and irrigation play a crucial role in determining yield.  

---

## **CONCLUSION**  
1. **Rainfall and soil type significantly influence crop yield.**  
2. **Random Forest Regression provides accurate yield predictions across regions.**  
3. Future improvements should include **additional factors (crop rotation, soil pH, disease control).**  

---

## **HOW TO RUN THE PROJECT**  
### **1. Prerequisites**  
Ensure you have Python installed along with required libraries:  
```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```
### **2. Clone the Repository**  
```bash
git clone https://github.com/yourusername/Crop-Yield-Prediction-Random-Forest.git
cd Crop-Yield-Prediction-Random-Forest
```
### **3. Run the Jupyter Notebook**  
```bash
jupyter notebook crop_yield_prediction_across_regions.ipynb
``` 
