# Logistic Regression: Probability-Based Classification

---

### Module Objective

This module solidifies the principles of probability modeling in machine learning. You will master the complete flow from raw scores to confident probabilities, recognizing that logistic regression is the foundational method for probabilistic binary and multi-class classification.

***

## I. The Principle of Probabilistic Modeling
While linear classifiers are effective, they only provide a hard decision boundary (Class 0 or Class 1). Logistic Regression solves this by introducing probability into the system.

### How it Works: The Three Steps
1.  **Linear Score ($Z$):** Calculate the weighted sum of features using the learned parameters ($\mathbf{W}, b$). $Z = \mathbf{W} \cdot \mathbf{X} + b$.
2.  **Sigmoiding:** Pass $Z$ through the Sigmoid function to get a probability $p$. This transforms the score into a measure of likelihood $[0, 1]$.
3.  **Thresholding (Prediction):** Use a threshold (typically $0.5$) on $p$ to assign the final discrete class label ($\hat{y}$).

## II. The Math Behind the Confidence

### 1. Linear Score ($Z$)
The score is calculated using linear algebra, exactly like in regression, but its physical meaning changes: it represents a **log-odds score**, not an expected value.

### 2. Sigmoid Function (Confidence Quantification)
$$p = \frac{1}{1 + e^{-Z}}$$
*   **Function:** It maps the continuous linear score $Z$ to a probability $p$.
*   **Interpretation:** This function provides model *confidence*. The closer $p$ is to 0 or 1, the more confident the model is. Being near $0.5$ means low confidence and suggests the sample lies close to the decision boundary.

### 3. Decision Boundary & Hyperplanes
The decision boundary is conceptually a hyperplane that separates the feature space into distinct regions corresponding to different classes. The linear score $Z=0$ represents the perfect separation point (the boundary).

## III. Multi-Class Extension and Cost Function
*   **Multi-Output:** For classification with $>2$ classes, we use one-hot encoding for labels and calculate a probability for every class using specialized loss functions like **Cross-Entropy Loss**.
*   **The Training Goal:** The entire optimization process aims to adjust $\mathbf{W}$ and $b$ so that the highest predicted probabilities align perfectly with the true class labels.

### Comparison Summary: Regression vs. Classification

| Feature | Linear Regression (MSE) | Logistic Regression (Cross-Entropy) |
| :--- | :--- | :--- |
| **Output $\hat{y}$** | Continuous value ($\mathbb{R}$) | Probability $(0, 1)$ |
| **Loss Function** | Mean Squared Error (MSE) | Cross-Entropy Loss |
| **Goal** | Predict the best continuous fit. | Maximize the probability of the correct class. |
| **Interpretation** | How far off is the prediction? | How certain is the prediction? |