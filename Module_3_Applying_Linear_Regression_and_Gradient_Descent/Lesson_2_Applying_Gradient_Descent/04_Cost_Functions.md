# Understanding Cost Functions (Loss Metrics) in PyTorch

---

### Module Objective

By mastering loss function selection, you will be able to:
*   Explain that the loss function is the quantitative measure that guides the entire model optimization process.
*   Differentiate between common losses used for various tasks (MSE for Regression, Cross-Entropy for Classification).
*   Understand the critical trade-off between MSE and MAE regarding outlier sensitivity.
*   Recognize that minimizing the loss is the ultimate objective of gradient descent.

***

## I. The Role of Cost Functions

### What They Do: Quantifying Error
A cost function $\mathcal{L}$ evaluates a model's performance by comparing its predicted output ($\hat{y}$) against the true target value ($y$). It converts prediction error into a single, continuous number—the **Cost Value**.

**Mechanism:**
1.  The model makes a prediction based on its current parameters $(\mathbf{W}, b)$.
2.  The loss function calculates $\text{Loss} = \mathcal{L}(\hat{y}, y)$.
3.  This resulting loss value is the metric that optimization algorithms (like Gradient Descent) attempt to minimize.

### The Loss Landscape
*   Conceptually, training involves navigating a high-dimensional **loss surface**. Every point on this surface represents a set of parameters $(\mathbf{W}, b)$ and its corresponding loss.
*   The goal of gradient descent is to find the lowest point (the minimum) on this surface, as that location corresponds to the optimal model parameters.

## II. Comparing Regression Loss Functions

When predicting continuous values (Regression), MSE and MAE are the two most common choices:

| Metric | Formula | Penalty Mechanism | Key Characteristic | When to Use |
| :--- | :--- | :--- | :--- | :--- |
| **Mean Squared Error (MSE)** | $\frac{1}{N} \sum (\hat{y} - y)^2$ | Squares the error, amplifying large mistakes significantly. | Smoothly differentiable; strong penalty on outliers. | When minimizing large errors is critical, and data noise is minimal. |
| **Mean Absolute Error (MAE)** | $\frac{1}{N} \sum \|\hat{y} - y\|$ | Treats all errors linearly—the magnitude of the error persists regardless of size. | Robust to outliers; treats all errors proportionally. | When the dataset is known to contain many extreme or noisy observations. |

### The Outlier Trade-off
The key difference lies in their sensitivity to **outliers** (extreme data points):

*   **MSE:** Because it squares the error, an outlier ($10$ units of error) contributes $100$ to the loss, forcing the model to aggressively shift parameters just to account for that single point.
*   **MAE:** An outlier of $10$ units only contributes $10$. This makes MAE more **robust** when dealing with highly noisy data, preventing a few outliers from dominating the entire training process.

## III. Conclusion: The Feedback Loop

The loss function is not just an evaluation tool; it is the primary feedback mechanism that drives learning:

$$\text{Model} \xrightarrow{\text{Parameters}} \text{Prediction} \xrightarrow{\text{Loss Function}} \text{Error Value} \xrightarrow{\text{Backpropagation}} \text{Gradients} \xrightarrow{\text{Optimizer}} \text{Updated Parameters}$$

By correctly selecting and minimizing the loss function, we ensure that the model's parameters converge to an optimal state that accurately reflects the underlying relationship in the data.