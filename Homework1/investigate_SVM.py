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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

print("SVM Investigation Started")