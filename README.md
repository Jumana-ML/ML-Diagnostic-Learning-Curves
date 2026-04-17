# Learning Curves Diagnostic: Telecom Churn Model

This project focuses on diagnosing the performance of a Logistic Regression model used to predict customer churn in a telecom dataset. Using `sklearn.learning_curve`, we analyze the bias-variance tradeoff to determine the best strategy for model improvement.

## 📊 The Diagnostic Plot
The following plot shows the F1-score as the training set size increases. It serves as a visual tool to understand whether the model needs more data or more complexity.

## 🔍 Written Analysis

### 1. Bias vs. Variance Diagnosis
Based on the learning curves, the model is primarily suffering from **High Bias (Underfitting)**.

*   **Evidence:** Both the training score (red line) and the cross-validation score (green line) converge at a relatively low F1-score (approximately 0.35–0.38). 
*   In a high-bias scenario, the model is too simple to capture the underlying patterns in the data, leading to poor performance even on the training set itself.

### 2. Would More Data Help?
**No.** Collecting more data is unlikely to significantly improve the validation performance.

*   **Evidence:** The curves have flattened out (reached a plateau) after around 800 training samples. The gap between the training and validation scores is very narrow, indicating that the model has already learned as much as its current architecture allows. Adding more rows will not change the fact that a linear model cannot fit these complex non-linear relationships.

### 3. Model Complexity & Recommendations
Increasing model complexity is the recommended next step. 

*   **Why?** Since the model is underfitting, we need to increase its capacity to learn. Logistic Regression (a linear classifier) is likely too restrictive for the churn dataset.

**Recommended Actions:**
*   **Feature Engineering:** Adding polynomial features or interaction terms to help the linear model understand non-linear relationships.
*   **Advanced Models:** Switching to more flexible, non-linear models such as **Random Forest**, **XGBoost**, or Support Vector Machines (SVM).
*   **Hyperparameter Tuning:** Reducing the regularization strength (e.g., increasing the `C` parameter in Logistic Regression) to allow the model to fit the training data more closely.

## 📈 Metric Justification
We used the **F1-Score** as our evaluation metric instead of Accuracy. This is because the telecom churn dataset has a significant class imbalance (~16% churn). Accuracy can be misleading in such cases; a model that predicts "No Churn" for everyone would still be 84% accurate but useless. F1-score provides a better balance between Precision and Recall.