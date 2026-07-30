# Linear Regression Training Cycle in PyTorch

---

### Module Objective

This module integrates all previous concepts—linear equations, cost functions, gradients, and optimization—into a coherent, working training loop for a simple predictive model. You will master the end-to-end process of teaching parameters to fit data.

***

## I. The Theory: Linear Modeling Foundation

### The Model Equation
Linear Regression assumes that the relationship between input ($\mathbf{X}$) and output ($\mathbf{Y}$) can be modeled by a straight line:
$$\hat{y} = w \cdot x + b$$
*   **Goal of Training:** To find the optimal values for $w$ (weight/slope) and $b$ (bias/intercept).

### The Optimization Objective
The training process is fundamentally an effort to minimize the **Cost Function** ($\text{MSE}$), which acts as a surface that must be minimized using gradient descent.

*   **Mechanism:** By calculating how much the loss changes relative to $w$ and $b$, we gain directional advice on how to adjust these parameters.
*   **Loss Landscape:** The goal is to move parameters down the cost landscape until the bottom (the global minimum) is reached—the point where the prediction error cannot be reduced further for that dataset.

## II. PyTorch Implementation: The Training Loop

The training process must follow a strict, iterative sequence of steps performed over multiple **epochs**.

### 1️. Initialization
*   **Parameters:** Initialize $\mathbf{w}$ and $b$ as trainable tensors by setting `requires_grad=True`. This signals to AutoGrad that these values must be tracked for derivative computation.
*   **Loss Function:** Select the appropriate loss metric (e.g., MSE).

### 2️. Forward Pass & Loss Calculation
*   **Prediction ($\hat{y}$):** Compute $\mathbf{w} \cdot \mathbf{X} + b$. This is the model's current best guess.
*   **Cost:** Calculate $\text{Loss}(\hat{y}, y)$ by comparing predictions to actual values.

### 3️. Backward Pass (Gradient Computation)
*   Calling `.backward()` triggers AutoGrad. It computes **$\frac{\partial \text{Loss}}{\partial w}$ and $\frac{\partial \text{Loss}}{\partial b}$**. These gradients quantify the error contribution of each parameter.

### 4️. Parameter Update (Optimization)
The optimizer uses these gradients to adjust parameters, taking a step *downhill* on the cost surface:
$$\mathbf{W}_{\text{new}} = \mathbf{W}_{\text{old}} - (\eta \cdot \frac{\partial \text{Cost}}{\partial W})$$

### 5️. Gradient Reset
After every update, it is crucial to call `optimizer.zero_grad()` to clear the stored gradients ($\mathbf{w}.\text{grad}$ and $b.\text{grad}$). Failing to do this will accumulate old gradients into new ones, leading to incorrect parameter updates.

> **The Training Cycle:** $\text{Predict} \to \text{Loss} \to \text{Backward} \to \text{Update} \xrightarrow{\text{Repeat}} \text{Converge}$