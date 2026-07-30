# Linear Regression Prediction in PyTorch

---

### Module Objective

By completing this module, you will be able to:
*   Understand linear regression as a prediction model for continuous outcomes.
*   Implement the forward pass manually and using built-in PyTorch modules (`nn.Linear`).
*   Structure predictions using custom classes that inherit from `nn.Module`, which is standard practice in deep learning.
*   Distinguish between manual parameter handling, module initialization, and structured prediction pipelines.

***

## I. Linear Regression: The Prediction Model

### Concept
Linear regression aims to model the relationship between an independent variable ($\mathbf{X}$) and a dependent variable ($\mathbf{Y}$) using the simplest linear equation:
$$\hat{y} = \text{Weight}(\mathbf{w}) \cdot \mathbf{x} + \text{Bias}(b)$$

*   **Prediction Step (Forward Pass):** Once training is complete, prediction is straightforward. We simply feed a new input $\mathbf{X}_{\text{new}}$ through the model's equation to calculate the predicted output $\hat{y}$.
*   $\mathbf{w}$ (**Weight** or Slope): Determines the steepness of the relationship.
*   $b$ (**Bias** or Intercept): Shifts the entire line up or down.

## II. Implementation Approaches in PyTorch

There are three ways to execute a linear prediction in PyTorch, increasing in complexity and professional standard:

### 1. Manual Implementation (Fundamentals)
This approach manually defines $\mathbf{w}$ and $b$ as tensors and implements the equation directly.
*   **Process:** The function simply computes `y_hat = w * X + b`.
*   **Benefit:** Provides deep insight into exactly how linear algebra is applied in practice.
*   **Use Case:** Ideal for initial understanding and debugging small, controlled examples.

### 2. Using `torch.nn.Linear` (The Standard Way)
PyTorch provides the built-in `nn.Linear` module, which abstracts away the manual handling of weights and biases.
*   **Initialization:** You specify two values: the number of input features (`input_features`) and the desired output dimension (`output_features`). The layer automatically creates and initializes the optimal $\mathbf{w}$ and $b$ parameters internally.
*   **Prediction:** Prediction is as simple as calling the module on the input tensor: `output = linear_layer(X)`.

### 3. Custom Model Implementation (The Professional Standard)
For building any deep learning model, it's best practice to define a custom class that inherits from $\mathbf{nn.Module}$. This allows you to combine multiple layers and operations into one cohesive unit.

*   **`__init__(self)`:** The constructor defines the components of your model (e.g., `self.linear_layer = nn.Linear(..., ...)`).
*   **`forward(self, input)`:** This method explicitly defines *how* data flows through the network. When you call `model(input)`, PyTorch automatically executes this `forward` method.

> **State Dictionary:** The parameters ($\mathbf{w}, b$) learned by the model are stored within its `state_dict()`, which is the dictionary that gets saved and loaded when saving/loading a trained model.

## Summary of Prediction Flow

| Stage | Method | Code Example (Concept) | Output $\hat{y}$ Represents |
| :--- | :--- | :--- | :--- |
| **Manual** | Direct calculation using `torch.tensor()` | `w * X + b` | The direct linear prediction for the input $X$. |
| **Standard PyTorch** | Using `nn.Linear` module | `linear(X)` | The predicted output, ready for further layers. |
| **Advanced** | Subclassing `nn.Module` | `forward_pass(X)` | The complete prediction from a complex multi-layer architecture. |