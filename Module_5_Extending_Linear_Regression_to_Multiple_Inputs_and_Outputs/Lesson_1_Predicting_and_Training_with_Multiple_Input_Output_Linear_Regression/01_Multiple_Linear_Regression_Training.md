# Multiple Linear Regression: Vectorized Prediction

---

### Module Objective

This module scales our understanding from single-variable linear regression to multiple variables, demonstrating how vector mathematics and PyTorch modules generalize the process. You will understand that model parameters become vectors, but the core training logic remains the same.

***

## I. The Conceptual Leap: From Scalar to Vector Parameters

### Multiple Linear Regression
When predicting an output ($\hat{y}$) using multiple input features ($\mathbf{x}$), the single weight ($w$) is replaced by a **weight vector** ($\mathbf{W}$).

*   **Equation:** $\hat{y} = \mathbf{W} \cdot \mathbf{X} + b$
*   **Structure:** If the input has $d$ features, the model must learn:
    1.  A $d$-dimensional weight vector ($\mathbf{W}$)—one weight for each feature.
    2.  A single bias ($b$).

### Training on Multiple Variables
The cost function remains MSE, but the parameters being optimized are now vectors. Gradient descent performs updates not on a scalar value, but on *each element* of the parameter vector ($\mathbf{W}$ and $b$) based on its individual contribution to the total loss.

## II. PyTorch Implementation: The `nn.Linear` Layer

PyTorch handles this generalization seamlessly using the $\mathbf{nn.Linear}$ module.

### Role of `nn.Linear(input_features, output_features)`
The linear layer encapsulates the entire transformation: $\hat{y} = \text{Input} \cdot \text{Weights}^T + \text{Bias}$.
*   **Automatic Management:** By specifying the number of input features, `nn.Linear` automatically creates and manages the correct dimensionality for both the weight vector ($\mathbf{W}$) and the bias ($b$).

### The Training Loop (The Generalization)
The core loop structure remains constant regardless of the model complexity:

1.  **Forward Pass:** $\hat{\mathbf{y}} = \text{Model}(\mathbf{X}_{\text{batch}})$
2.  **Loss Calculation:** Calculate $L(\hat{\mathbf{y}}, y)$.
3.  **Backward Pass:** Compute gradients for all parameters in the model: `.backward()`.
4.  **Update Parameters:** Use the optimizer to apply gradient updates across **all** internal weight vectors and biases.

## III. The Power of Vectorization

The transition from single-variable to multi-variable models demonstrates the immense power of vectorization:

*   Instead of looping through features manually, PyTorch processes all $d$ input features simultaneously using matrix multiplication (the dot product of the entire $\mathbf{X}$ batch with the weight vector $\mathbf{W}$).
*   This means that changing one parameter ($\mathbf{w}_1$) is treated independently from another ($\mathbf{w}_2$), yet their combined effect determines the final output.

> **Summary:** Multiple linear regression generalizes the concept of a single slope ($y=wx+b$) into an entire hyperplane defined by a weight vector and a bias, allowing the model to capture complex interactions between many features while maintaining the core principles of gradient descent optimization.