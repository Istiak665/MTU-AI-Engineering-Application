"""
Homework 1: Investigation of SVM
Course: AI Engineering Application
Student: Istiak Ahammed
"""

# =============================
# 1. Import Libraries
# =============================

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# =============================
# 2. Load Dataset
# =============================

data = load_breast_cancer()

X_all = pd.DataFrame(data.data, columns=data.feature_names)
y_all = pd.Series(data.target, name="target")

print("Total samples:", X_all.shape[0])
print("Total features:", X_all.shape[1])
print("Target classes (0=malignant, 1=benign):", y_all.value_counts().to_dict())

# =============================
# 3. Feature Analysis & Selection
# =============================

"""
Goal:
Select 4–6 most relevant features.

Strategy:
We compute the correlation between each feature and the target variable.
Then we select the top 5 features with highest absolute correlation.
"""

# Combine features and target into one DataFrame for correlation analysis
df = X_all.copy()
df["target"] = y_all

# Compute correlation matrix
correlation_matrix = df.corr()

# Extract correlation values with respect to target
target_correlation = correlation_matrix["target"].drop("target")

# Sort features by absolute correlation value (descending)
sorted_features = target_correlation.abs().sort_values(ascending=False)

print("\nFeature correlations with target (absolute values):")
print(sorted_features.head(10))

# Select top 5 most correlated features
selected_feature_names = sorted_features.head(5).index.tolist()

print("\nSelected top 5 features based on correlation:")
print(selected_feature_names)

# Create new dataset with only selected features
X_selected = X_all[selected_feature_names]
y_selected = y_all
