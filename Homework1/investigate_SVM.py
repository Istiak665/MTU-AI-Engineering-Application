"""
Homework 1: Investigation of SVM
Course: AI Engineering Application
Student: Istiak Ahammed
"""

# =============================
# 1. Import Libraries
# =============================
import os
import matplotlib.pyplot as plt
import seaborn as sns
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

# =============================
# 4. Train-Test Split
# =============================

"""
Goal:
Split the selected dataset into training (80%) and testing (20%).

Important:
- stratify=y_selected ensures class distribution remains balanced.
- random_state=42 ensures reproducibility.
"""

X_train, X_test, y_train, y_test = train_test_split(
    X_selected,
    y_selected,
    test_size=0.2,        # 20% for testing
    random_state=42,      # ensures same split every run
    stratify=y_selected   # keeps class distribution consistent
)

print("\nTrain/Test Split Summary:")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Number of features used:", X_train.shape[1])

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


# =============================
# 5. Feature Scaling
# =============================

"""
Goal:
Standardize the feature values.

Why scaling is important?
SVM is a distance-based algorithm.
Features with larger numeric ranges can dominate smaller ones.
Therefore, we standardize features to have:
    - Mean = 0
    - Standard deviation = 1

Important:
We fit the scaler ONLY on training data.
Then we transform both training and testing data.
This avoids data leakage.
"""

# Initialize scaler
scaler = StandardScaler()

# Fit only on training data
X_train_scaled = scaler.fit_transform(X_train)

# Use same transformation on test data
X_test_scaled = scaler.transform(X_test)

print("\nFeature Scaling Completed.")
print("Scaled Training Data Shape:", X_train_scaled.shape)
print("Scaled Testing Data Shape:", X_test_scaled.shape)

# Optional sanity check: mean and std of training set
print("\nMean of first scaled feature (train):", np.mean(X_train_scaled[:, 0]))
print("Std of first scaled feature (train):", np.std(X_train_scaled[:, 0]))

# =============================
# 6. SVM Model Training (Multiple Kernels)
# =============================

"""
Goal:
Train SVM models using different kernel functions.

We will test:
    - Linear
    - Polynomial
    - RBF (Gaussian)
    - Sigmoid

For each kernel:
    1. Train the model
    2. Make predictions on test set
    3. Store predictions for evaluation
"""

# Define kernels to test
kernels = ["linear", "poly", "rbf", "sigmoid"]

# Dictionary to store predictions for each kernel
predictions = {}

print("\nTraining SVM models with different kernels...\n")

for kernel in kernels:
    # Initialize SVM model with current kernel
    model = SVC(kernel=kernel, random_state=42)

    # Train the model
    model.fit(X_train_scaled, y_train)

    # Predict on test data
    y_pred = model.predict(X_test_scaled)

    # Store predictions
    predictions[kernel] = y_pred

    print(f"Kernel '{kernel}' training completed.")

# =============================
# 7. Performance Evaluation
# =============================

"""
Goal:
Evaluate performance of each SVM kernel using:

    - Confusion Matrix (saved as PNG)
    - Accuracy
    - Precision
    - Recall
    - F1 Score

All metrics will be saved into a CSV file.
"""

# Create output directory if not exists
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# List to store metric results
results = []

print("\nEvaluating Models...\n")

for kernel in kernels:
    y_pred = predictions[kernel]

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Save metrics in results list
    results.append({
        "Kernel": kernel,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1
    })

    # Plot confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {kernel.upper()} Kernel")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Save figure
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{kernel}.png"))
    plt.close()

    print(f"Kernel '{kernel}' evaluation completed.")

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Display comparison table
print("\nModel Performance Comparison:")
print(results_df)

# Save results to CSV
csv_path = os.path.join(output_dir, "svm_kernel_comparison.csv")
results_df.to_csv(csv_path, index=False)

print(f"\nMetrics saved to: {csv_path}")
print("Confusion matrices saved inside 'results/' folder.")

# =============================
# 8. Conclusion
# =============================

"""
Goal:
Interpret the performance results and identify the best-performing kernel.
"""

# Identify best kernel based on highest accuracy
best_model = results_df.loc[results_df["Accuracy"].idxmax()]

best_kernel = best_model["Kernel"]
best_accuracy = best_model["Accuracy"]

print("\n=============================")
print("FINAL CONCLUSION")
print("=============================")

print(f"\nBest Performing Kernel: {best_kernel.upper()}")
print(f"Highest Accuracy Achieved: {best_accuracy:.4f}")

print("\nDetailed Performance of Best Model:")
print(f"Precision: {best_model['Precision']:.4f}")
print(f"Recall: {best_model['Recall']:.4f}")
print(f"F1 Score: {best_model['F1_Score']:.4f}")

"""
==============================
Result Interpretations
==============================

In this investigation, Support Vector Machine (SVM) models were evaluated
using four different kernel functions: Linear, Polynomial, RBF, and Sigmoid.

Five most relevant features were selected based on highest correlation
with the target variable to reduce dimensionality while preserving
discriminative information.

Among the tested kernels, the Linear kernel achieved the highest accuracy
(94.74%), along with strong Precision (97.14%), Recall (94.44%), and
F1-score (95.77%).

This suggests that the selected features allow the data to be approximately
linearly separable in the reduced feature space.

Although RBF and Polynomial kernels also performed competitively,
the Linear kernel provided the best overall balance of performance
metrics for this dataset.

Therefore, for the selected features and dataset configuration,
the Linear SVM is the most suitable model.
"""

