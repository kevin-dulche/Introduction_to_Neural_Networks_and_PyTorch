# Understanding Differentiation in PyTorch

---

### Module Objective

After completing this module, you will be able to:
*   Explain the fundamental role of automatic differentiation (AutoGrad) in guiding model training.
*   Distinguish between gradients, partial derivatives, and their practical use in parameter updates.
*   Trace the flow of derivative computation using the computational graph and the chain rule.

***

## I. The Need for Differentiation: Why Train Models?

**Goal:** To train a neural network to minimize a **Loss Function** (which measures error).
$$\text{Minimize Loss} = \mathcal{L}(\hat{y}, y)$$
Where $\hat{y}$ is the prediction and $y$ is the true value.

**The Core Question:** How do we know how much to adjust each model weight ($\mathbf{W}$) or bias ($b$) to make the loss smaller? The answer lies in **calculating derivatives**. Gradients quantify exactly this: *how sensitive is the output (Loss) to small changes in a specific parameter?*

## II. Automatic Differentiation (`torch.autograd`)

PyTorch provides `AutoGrad`—a powerful system that handles complex differentiation automatically, allowing data scientists to focus on model architecture rather than calculus implementation.

### The Computational Graph
When operations are performed, PyTorch doesn't just compute the result; it records *how* the result was computed.
1.  **Recording:** When a tensor is created with `requires_grad=True`, PyTorch begins tracking every operation (multiplication, addition, etc.) that involves this tensor.
2.  **Graph Construction:** These recorded operations form the **Computational Graph**. This graph maps dependencies: *The output depends on these specific inputs via these sequence of mathematical operations.*

### The Backward Pass (`.backward()`)
After computing a loss $\mathcal{L}$, calling `.backward()` initiates the gradient calculation across the computational graph in reverse order (the backpropagation process). PyTorch automatically calculates $\frac{\partial \mathcal{L}}{\partial W}$ for every parameter $W$ that was tracked.

> **Key Principle:** Gradients are stored in the `.grad` attribute of the parameters, telling the optimization algorithm exactly how to move the weights to reduce the loss.

## III. Mathematical Concepts in Practice

### Partial Derivatives
*   **Concept:** Measures how a multi-variable function changes with respect to *one specific variable*, while holding all other variables constant.
*   **ML Context:** Since a loss function depends on hundreds or thousands of parameters (weights and biases), we compute the gradient using partial derivatives: $\frac{\partial \mathcal{L}}{\partial w_{ij}}$. This isolates the contribution of every single parameter to the total error.

### The Chain Rule
*   **Concept:** For complex functions composed of multiple layers ($y = g(f(x))$), the derivative must be calculated sequentially by multiplying the derivatives at each intermediate step: $\frac{dy}{dx} = \frac{dg}{df} \cdot \frac{df}{dx}$.
*   **Mechanism in PyTorch:** The `AutoGrad` system automatically implements the chain rule. As the backward pass traverses the computational graph, it propagates gradients layer by layer, ensuring that the final gradient calculation for the initial input is accurate despite the complexity of the model.

## IV. The Training Cycle: Utilizing Gradients

The entire process connects these mathematical concepts into a practical loop:

1.  **Forward Pass:** Input $\mathbf{x} \xrightarrow{\text{Model}} \text{Prediction } \hat{y}$.
2.  **Calculate Loss:** Compute the error $L = \mathcal{L}(\hat{y}, y)$.
3.  **Backward Pass (Gradient Computation):** Call `.backward()` to compute $\frac{\partial L}{\partial \mathbf{W}}$ using AutoGrad and the Chain Rule.
4.  **Optimization Step:** The optimizer uses these gradients ($\mathbf{g}$) to update the parameters: $\mathbf{W}_{\text{new}} = \mathbf{W}_{\text{old}} - \eta \cdot \mathbf{g}$ (where $\eta$ is the learning rate).