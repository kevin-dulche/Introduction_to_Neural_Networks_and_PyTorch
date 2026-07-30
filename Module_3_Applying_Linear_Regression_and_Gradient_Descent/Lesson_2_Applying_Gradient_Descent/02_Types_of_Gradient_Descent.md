# Types of Gradient Descent: Optimizing Model Parameters

---

### Module Objective

This reading solidifies your understanding of parameter optimization by detailing the mechanics, trade-offs, and practical application of various gradient descent variants. By the end, you will be able to select the appropriate optimization strategy for any given dataset size and hardware constraint.

***

## I. The Core Mechanism: Parameter Updates

### Gradient Descent Review
The core concept remains consistent: find the lowest point on the loss landscape by iteratively stepping in the opposite direction of the gradient (the steepest decline).

**The Update Rule:**
$$\mathbf{W}_{\text{new}} = \mathbf{W}_{\text{old}} - (\text{Learning Rate} \times \text{Gradient})$$

### The Role of Hyperparameters
*   **Learning Rate ($\eta$):** Controls the step size. Must be tuned to balance fast convergence (high $\eta$) vs. stability/avoiding overshoot (low $\eta$).
*   **Stopping Criteria:** Defines when training stops (e.g., fixed epochs, loss plateau, validation performance degradation).

## II. Comparing Gradient Descent Variants

The difference between these three variants lies solely in **how many data points are used to calculate the gradient at each step.**

| Variant | Data Used Per Update | Calculation Frequency | Path Trajectory | Computational Cost/Efficiency | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Batch Gradient Descent (BGD)** | **Full Dataset ($\mathbf{N}$ samples)** | Once per epoch. | Smooth, accurate path toward the minimum. | High cost: Slow and memory-intensive for large $\mathbf{N}$. | Small datasets where precision is paramount. |
| **Stochastic Gradient Descent (SGD)** | **Single Sample ($1$ sample)** | Very frequent (updates every single example). | Highly noisy, zigzagging path. | Fast updates; suitable when BGD is impossible due to size. | Exploration of complex parameter spaces (sometimes used as a refinement step). |
| **Mini-Batch GD (The Industry Standard)** | Small Subset ($\mathbf{B}$ samples) | Frequent, but less frequent than SGD. | Stable enough path with manageable noise. | Optimal balance: Computationally efficient and stable. | Virtually all modern deep learning applications. |

## III. Epochs vs. Iterations (Tracking Progress)

Understanding these terms is key to monitoring training progress:

*   **Epoch:** One complete pass through the entire dataset.
    $$\text{Total Training} = \text{Multiple Epochs}$$
*   **Iteration:** A single parameter update step.
*   **Calculation:** $\text{Iterations per Epoch} = \frac{\text{Total Samples}}{\text{Batch Size}}$

> **Example:** If you have 1,000 samples and a batch size of 100:
> *   1 epoch requires $10$ iterations.
> *   If you train for $5$ epochs, you perform $50$ total iterations.