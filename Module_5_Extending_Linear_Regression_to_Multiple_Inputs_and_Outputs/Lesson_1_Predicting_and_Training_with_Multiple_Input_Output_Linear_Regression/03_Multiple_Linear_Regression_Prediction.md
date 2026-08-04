# Multiple Linear Regression: Scaling Up Predictions

---

### Module Objective

This module generalizes linear modeling from single-variable to multi-variable settings. You will master the mathematical requirements of high-dimensional prediction, understand how PyTorch's `nn.Linear` layer abstracts this complexity, and recognize that linear regression forms the indispensable bedrock for all subsequent complex neural networks.

***

## I. The Theory: Extending Linearity to Hyperplanes

### Multiple Linear Regression
When an output ($\hat{y}$) depends on multiple inputs ($x_1, x_2, \dots, x_d$), we replace the single slope with a **weight vector** ($\mathbf{W}$).

*   **Formula:** $\hat{y} = \mathbf{W} \cdot \mathbf{X} + b$
    $$\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_d x_d + b$$
*   **Geometric Interpretation:** Instead of a line (2D), the model creates a flat plane or **hyperplane** in $d+1$ dimensional space to best fit all data points.

### The Importance of Tensor Shapes
Correct tensor shaping is paramount for valid linear algebra operations:
*   **Single Sample Input:** $\mathbf{X}$ must have compatible dimensions with the weight vector $\mathbf{W}$. The dot product calculation requires that the number of columns in the input matrix equals the number of rows in the weight matrix.
*   **Batch Input (The Matrix):** When processing $N$ samples, the inputs are organized as an $N \times d$ matrix ($\mathbf{X}$).
*   **Prediction Output:** The final output will be an $N \times 1$ vector—a prediction for every sample in the batch.

## II. Implementation and Best Practices in PyTorch

### 1. Using `nn.Linear` (The Standard Way)
This class is designed precisely for the multiple linear regression transformation:
*   **Initialization:** Define `nn.Linear(in_features, out_features)`.
    *   `in_features`: Must match the number of columns in your input data ($\mathbf{X}$).
    *   `out_features`: The dimension of your target variable (typically 1 for simple regression).
*   **Prediction:** Passing the $\mathbf{X}$ tensor directly to `model(X)` executes the full linear transformation efficiently.

### 2. Custom Module Design (`nn.Module`)
For building complex models, we subclassing `nn.Module` is best practice:
*   The `__init__` method defines and initializes all internal layers (e.g., a single `nn.Linear` layer).
*   The `forward` method dictates the exact sequence of operations that transforms raw input into final predictions.

## III. Linear Regression as the Core Building Block

**Every modern neural network is fundamentally built by stacking linear transformations.**

When you build deep networks, a single layer might perform:
$$\text{Layer Output} = \mathbf{W} \cdot \mathbf{X} + b$$
...followed immediately by an **Activation Function** (like ReLU or Sigmoid). The activation function introduces the non-linearity needed to solve complex, real-world problems that linear models cannot handle.

Understanding how the simple `nn.Linear` layer performs this core transformation is key to understanding all advanced deep learning architectures.