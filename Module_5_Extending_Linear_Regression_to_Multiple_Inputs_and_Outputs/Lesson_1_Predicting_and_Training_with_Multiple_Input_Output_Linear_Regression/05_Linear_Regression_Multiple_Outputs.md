# Multi-Output Linear Regression: Expanding Prediction Scope

---

### Module Objective

This module teaches how to extend linear regression from predicting a single value (scalar) to predicting multiple values (vector). You will master the conceptual shift from weight vectors to weight matrices and understand the specialized tensor operations required for multi-output prediction.

***

## I. Theory: Scaling Up Prediction Outputs

### The Problem
Many real-world scenarios require more than one output. Example: Predicting both a house's *price* **and** its predicted market value in a second currency, given the same features (size, location).

### Vectorized Solution
Instead of running separate models for each output, we use a single mathematical structure that calculates all outputs simultaneously using matrix algebra.

*   **Input:** A vector $\mathbf{x}$ with $d$ features ($1 \times d$).
*   **Parameters:** The weights are grouped into a **Weight Matrix** ($\mathbf{W}$) of shape ($d \times m$), where $m$ is the number of outputs.
*   **Bias:** We use an expanded bias vector $\mathbf{b}$ of shape $(1 \times m)$.

### Prediction Equation (Matrix Form)
The prediction for all $m$ outputs ($\hat{\mathbf{y}}$) from one input sample is given by:
$$\hat{\mathbf{y}} = \mathbf{x} \cdot \mathbf{W} + \mathbf{b}$$

*   **Calculation:** The dot product of the single input row vector $\mathbf{x}$ with every column in $\mathbf{W}$.
    *   Each **column** of $\mathbf{W}$ is dedicated to the weights required for *one specific output*.

## II. Tensor Shape Management (The Key to Success)

Understanding dimensions is vital; improper shapes will cause computational failure.

| Component | Single Sample Shape | Batch Sample Shape | Purpose |
| :--- | :--- | :--- | :--- |
| **Input Features ($\mathbf{X}$)** | $(1, d)$ | $(\text{\# Samples}, d)$ | The input data matrix (Samples $\times$ Features). |
| **Weight Matrix ($\mathbf{W}$)** | $(d, m)$ | N/A | Maps $d$ inputs to $m$ outputs. ($d$: rows; $m$: columns). |
| **Bias Vector ($\mathbf{b}$)** | $(1, m)$ | N/A | An offset value applied across all $m$ outputs. |
| **Output Prediction ($\hat{\mathbf{y}}$)** | $(1, m)$ | $(\text{\# Samples}, m)$ | The final matrix of predictions (Samples $\times$ Outputs). |

## III. PyTorch Implementation (`nn.Linear`)

PyTorch automatically handles the complex weight matrix setup:

*   **Implementation:** By specifying `out_features` in `nn.Linear(in_features, out_features)`, we tell the layer to create a weight matrix ($\mathbf{W}$) with the correct dimensions ($d \times m$) and an appropriate bias vector of size $m$.
*   **Benefit:** This single module handles all the necessary vector and matrix arithmetic internally, simplifying development.

### Conclusion: The Unified Framework
The ability to predict multiple outputs simultaneously is not a specialized function; it's simply the natural extension of linear algebra—treating the output as a multi-dimensional prediction vector. By using `nn.Linear`, PyTorch abstracts away the complex weight matrix management, allowing us to focus on building robust and scalable models.