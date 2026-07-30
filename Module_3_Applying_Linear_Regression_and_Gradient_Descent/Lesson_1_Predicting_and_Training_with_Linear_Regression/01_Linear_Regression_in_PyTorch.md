# Linear Regression Training with PyTorch

---

### Module Objective

After completing this module, you will be able to:
*   Understand the core concept of linear regression as a predictive function.
*   Implement a full training loop in PyTorch using gradient descent principles.
*   Apply the Mean Squared Error (MSE) loss function to measure model error.
*   Master the use of `requires_grad=True` and the `.backward()` method to compute gradients correctly.

***

## I. Linear Regression: The Foundation Model

### What is it?
Linear regression is one of the simplest, yet most powerful, machine learning methods used to model the relationship between a set of input features ($\mathbf{x}$) and a continuous target value ($\mathbf{y}$).

### The Core Equation (Simple Linear Case)
The prediction ($\hat{y}$) is based on finding the optimal slope ($w$) and intercept (bias $b$):
$$\hat{y} = w \cdot x + b$$
*   **Features ($\mathbf{x}$):** The input data, represented as a tensor.
*   **Weights ($\mathbf{w}$):** The model's learned parameter (the slope).
*   **Bias ($b$):** An offset parameter that shifts the line vertically.

### Dealing with Noise
Real-world data is never perfect. Small variations are modeled as **noise**, which we typically assume follows a Gaussian distribution. Despite this noise, our goal remains finding the single straight line that minimizes the overall error relative to the true underlying pattern.

## II. Training Workflow: The Gradient Descent Cycle

Training involves an iterative process where the model repeatedly adjusts its parameters ($\mathbf{w}$ and $b$) to minimize the prediction error.

### Step 1: Defining the Model Parameters
To ensure PyTorch can track which values must be optimized, we initialize $\mathbf{w}$ and $b$ with `requires_grad=True`. This tells AutoGrad that these tensors are trainable parameters.

$$\text{Model Parameters:} \quad w = \text{torch.tensor}(\dots, \text{requires\_grad}=True)$$

### Step 2: Forward Pass & Loss Calculation
1.  **Prediction:** Calculate the predicted output $\hat{y}$ using the current weights and biases.
2.  **Loss Function (MSE):** The Mean Squared Error (MSE) calculates the average squared difference between predictions ($\hat{y}$) and targets ($y$):
    $$\text{Loss} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$

### Step 3: Backward Pass (Gradient Computation)
*   **Action:** Calling `loss.backward()` triggers the magic. PyTorch traverses the computational graph backward from the loss, automatically calculating $\frac{\partial \text{Loss}}{\partial w}$ and $\frac{\partial \text{Loss}}{\partial b}$.
*   **Result:** The gradients are stored in `w.grad` and `b.grad`. These values quantify: *If I change $w$ slightly, how much does the loss increase/decrease?*

### Step 4: Parameter Optimization (Gradient Descent)
The optimizer uses the calculated gradients to update the parameters iteratively:
$$\mathbf{W}_{\text{new}} = \mathbf{W}_{\text{old}} - (\text{Learning Rate} \cdot \frac{\partial L}{\partial W})$$

**The Training Loop:** This sequence (Forward $\to$ Loss $\to$ Backward $\to$ Update) repeats for many **epochs** (full passes over the dataset). Successful training is indicated by a consistently decreasing loss value, showing that the model is converging toward an optimal fit.