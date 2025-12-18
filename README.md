---

# Rainfall Prediction with Random Forest Classifier

This project predicts the likelihood of rainfall using a **Random Forest Classifier**. We leverage historical weather data (such as temperature, humidity, wind speed, etc.) to determine if it will rain the following day. The goal is to build an accurate model using machine learning techniques, and the process involves data collection, preprocessing, exploratory data analysis (EDA), model training, and fine-tuning for optimal performance.

## Table of Contents

1. Project Overview
2. Getting Started
3. Dependencies
4. Data Collection and Preprocessing

   * Handling Missing Values
   * Feature Embedding
5. Exploratory Data Analysis (EDA)
6. Modeling

   * Train-Test Split
   * Random Forest Classifier
   * Hyperparameter Tuning with GridSearchCV
7. Results and Evaluation
8. Conclusion and Next Steps

---

## Project Overview

Predicting rainfall is important for sectors like agriculture, disaster management, and urban planning. In this project, we use a **Random Forest Classifier**, a powerful ensemble learning model, to predict whether it will rain the next day based on weather data. The model is trained and evaluated through a systematic process involving data preprocessing, feature engineering, and hyperparameter optimization.

---

## Getting Started

To replicate this project, you will need the following:

1. **Python 3.x** installed
2. The required dependencies (listed below)
3. A dataset containing daily weather data

Clone this repository and run the code from your local machine to start experimenting!

---

## Dependencies

This project relies on several popular Python libraries. Install them by running:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Additionally, you might want to install **XGBoost** for potential model comparisons:

```bash
pip install xgboost
```

---

## Data Collection and Preprocessing

### Data Collection

The dataset used in this project consists of daily weather records, including features such as temperature, humidity, wind speed, and atmospheric pressure. The target variable is whether or not it rained the following day (`Rain Tomorrow`).

### Data Preprocessing

Before building the model, we need to clean and prepare the data. This involves the following steps:

#### Handling Missing Values

Missing values can negatively impact model performance. In this project, we use simple imputation strategies or remove columns with too many missing values. For numeric columns, we use the median or mean, and for categorical variables, the mode is used.

```python
import pandas as pd

# Load the dataset
data = pd.read_csv('weather_data.csv')

# Check for missing values
print(data.isnull().sum())

# Impute missing values
data['Temperature'].fillna(data['Temperature'].median(), inplace=True)
data['Humidity'].fillna(data['Humidity'].mean(), inplace=True)
data['Rain Tomorrow'].fillna(data['Rain Tomorrow'].mode()[0], inplace=True)
```

#### Feature Embedding

Feature engineering might involve transforming categorical features into numerical representations. For example, if a feature is categorical (e.g., "wind direction"), we can use **one-hot encoding** to convert it into numerical format.

```python
# Example of encoding categorical variables
data = pd.get_dummies(data, columns=['Wind Direction'], drop_first=True)
```

---

## Exploratory Data Analysis (EDA)

EDA is the process of analyzing the dataset to understand its structure and identify key patterns, distributions, and correlations. During EDA, we will:

* Visualize distributions of key variables using histograms and boxplots.
* Examine correlations between features using heatmaps.
* Identify any outliers that could affect model performance.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Visualizing temperature distribution
sns.histplot(data['Temperature'], kde=True)
plt.title('Temperature Distribution')
plt.show()

# Correlation matrix heatmap
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()
```

---

## Modeling

### Train-Test Split

To evaluate the model's performance, we split the data into **training** and **test** sets. Typically, 80% of the data is used for training and 20% for testing.

```python
from sklearn.model_selection import train_test_split

# Features (X) and Target (y)
X = data.drop(columns=['Rain Tomorrow'])
y = data['Rain Tomorrow']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Random Forest Classifier

We use a **Random Forest Classifier** for this binary classification task. It works by creating multiple decision trees during training and outputting the most common class as the prediction.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize and train the model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))
```

### Hyperparameter Tuning with GridSearchCV

To improve the model's performance, we tune its hyperparameters using **GridSearchCV**. GridSearchCV systematically searches for the best combination of parameters to maximize model performance.

```python
from sklearn.model_selection import GridSearchCV

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Apply GridSearchCV to find the best parameters
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters found by GridSearchCV
print(f'Best hyperparameters: {grid_search.best_params_}')

# Evaluate the best model
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)

# Evaluate optimized model
print(f'Optimized Accuracy: {accuracy_score(y_test, y_pred_best)}')
print(classification_report(y_test, y_pred_best))
```

---

## Results and Evaluation

The performance of the model is evaluated using various metrics, such as:

* **Accuracy**: The percentage of correct predictions.
* **Precision**: The proportion of true positive predictions relative to all positive predictions.
* **Recall**: The proportion of true positives relative to all actual positives.
* **F1-Score**: The harmonic mean of precision and recall, offering a balanced measure of both.

Here’s an example of the evaluation output:

```
Accuracy: 0.85
Precision: 0.84
Recall: 0.86
F1-Score: 0.85
```

---

## Conclusion and Next Steps

This project successfully predicts rainfall using a **Random Forest Classifier**, and we've shown how to preprocess the data, explore its features, and fine-tune the model for improved accuracy.

### Next Steps:

* **Model Comparisons**: Test other machine learning models (e.g., Support Vector Machines, XGBoost) to compare performance.
* **Feature Expansion**: Add more features (e.g., past rainfall, geographical data) to enhance predictions.
* **Model Deployment**: Deploy the model into a web application or an API for real-time predictions.

We encourage you to experiment with different datasets, explore more advanced feature engineering techniques, and deploy the model to make it accessible to users.

---
