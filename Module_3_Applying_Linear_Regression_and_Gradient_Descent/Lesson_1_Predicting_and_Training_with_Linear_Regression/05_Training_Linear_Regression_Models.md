# Linear Regression Training Fundamentals in PyTorch

---

### Module Objective

By completing this module, you will be able to:
*   Define the concept of supervised learning as it applies to predicting continuous values (regression).
*   Understand how datasets and noise assumptions influence the goal of model training.
*   Utilize Mean Squared Error (MSE) Loss to quantitatively measure prediction error.
*   Recognize that parameter optimization is an iterative process focused on minimizing this loss function.

***

## I. Defining Linear Regression Training

### The Goal: Parameter Estimation
Training a linear regression model means finding the optimal parameters ($\mathbf{w}$ and $b$) that define the straight line which best fits the observed relationship between $\mathbf{X}$ (predictor/input) and $\mathbf{Y}$ (target/output).

*   **Model Function:** $\hat{y} = w \cdot x + b$
*   **Input Data ($\mathbf{X}, \mathbf{Y}$):** The data set must consist of labeled pairs ($\mathbf{x}_i, y_i$). Each pair acts as a guide for the model's learning process.

### Noise Assumption
In reality, data is messy. We assume that any observed deviation from the perfect straight line is due to **noise** (measurement errors, unknown factors). This assumption allows the model to ignore minor fluctuations and focus on learning the underlying systemic relationship.

## II. Measuring Error: The Loss Function

To quantify how "bad" a set of parameters ($\mathbf{w}, b$) are, we use a Loss Function.

### Mean Squared Error (MSE)
*   **Function:** MSE calculates the average squared difference between predicted values ($\hat{y}$) and actual target values ($y$).
    $$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$
*   **Why Squaring?** Squaring the error has two key effects:
    1.  It penalizes large errors much more heavily than small ones.
    2.  It eliminates negative signs, ensuring that positive and negative deviations contribute to increasing the loss equally.
*   **Goal of Training:** The entire objective is to find the parameters $(\mathbf{w}, b)$ that minimize this MSE Loss, resulting in the **best-fitting line**.

## III. The Iterative Learning Process (Training Loop)

Model training is not a single calculation; it's an iterative optimization cycle driven by minimizing the loss using gradient descent.

1.  **Initialization:** Initialize parameters ($\mathbf{w}, b$) randomly, and enable gradient tracking (`requires_grad=True`).
2.  **Forward Pass (Prediction):** Input data $\mathbf{X}$ is fed through the current model parameters to generate predictions $\hat{\mathbf{y}}$.
3.  **Loss Calculation:** MSE Loss measures the difference between $\hat{\mathbf{y}}$ and $\mathbf{Y}$.
4.  **Backward Pass (Gradient Descent):** The `.backward()` function computes the gradients ($\frac{\partial \text{Loss}}{\partial w}, \frac{\partial \text{Loss}}{\partial b}$) for all parameters. These gradients tell us the direction of steepest ascent in the loss landscape.
5.  **Parameter Update:** The optimizer uses these gradients to update the parameters by taking a step *opposite* the gradient direction (the steepest descent), thus reducing the loss.

> **Concept Summary:** Training is an endless loop: Predict $\to$ Calculate Error $\to$ Find Direction of Error Reduction $\to$ Adjust Parameters $\to$ Repeat.