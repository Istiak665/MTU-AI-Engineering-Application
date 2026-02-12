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