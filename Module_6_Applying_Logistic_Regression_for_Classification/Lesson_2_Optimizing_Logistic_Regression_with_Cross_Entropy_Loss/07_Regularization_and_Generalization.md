# Regularization and Generalization: Building Robust Models

---

### Module Objective

This reading is a deep dive into model robustness. You will learn how to systematically control complexity and improve generalization by selecting the appropriate regularization technique, ensuring your model performs reliably on real-world data.

***

## I. The Core Problem: Overfitting vs. Underfitting
*   **Overfitting:** The model learns the training data too well (including noise), resulting in high training accuracy but low performance on new data.
*   **Underfitting:** The model is too simple or constrained, failing to capture even the underlying patterns in the training data.

## II. Regularization Techniques (Controlling Complexity)
Regularization techniques add a penalty term to the cost function that discourages overly complex models.

### 1. Weight Regularization ($\mathbf{L1}$ and $\mathbf{L2}$)
These are penalties applied directly to the magnitude of the model's weights:

*   **L2 Regularization (Weight Decay):** Adds a penalty proportional to the *square* of the weights ($\lambda \sum w^2$). It gently shrinks all weights towards zero, distributing importance across all features.
    *   **Effect:** Produces stable, smooth models; prevents any single weight from becoming too large.
    *   **Ideal For:** General deep learning stability (most common choice).
*   **L1 Regularization:** Adds a penalty proportional to the *absolute value* of the weights ($\lambda \sum |w|$).
    *   **Effect:** Has an inherent mechanism for **automatic feature selection**, driving irrelevant weight values exactly to zero, simplifying the model structure.

### 2. Dropout (Stochastic Neuron Disabling)
Dropout is a powerful technique used specifically in neural network layers.
*   **Mechanism:** During *training*, it randomly sets a fraction ($p$) of neuron outputs to zero. This forces remaining neurons to not rely on any single feature or other specific neuron, building redundancy.
*   **Effect:** Improves model robustness and prevents co-adaptation between features, leading to better generalization. (Crucially, Dropout is **disabled** during inference.)

### 3. Normalization Techniques (Stabilizing Inputs)
These methods normalize the layer inputs to stabilize training dynamics:

*   **Batch Normalization (BatchNorm):** Normalizes activations across the *batch dimension*. It stabilizes inputs by ensuring that feature distributions remain consistent even if they shift during training—reducing "Internal Covariate Shift."
*   **Layer Normalization (LayerNorm):** Normalizes activations across the *feature dimension* for a single sample.
    *   **Advantage:** Independent of batch size, making it ideal and critical for sequence models (like Transformers) where mini-batches might be very small or variable.

## III. Advanced Training Controls

### Early Stopping
This is the most practical technique for generalization monitoring:
1.  Monitor $\text{Loss}_{\text{Validation}}$ throughout training.
2.  If Validation Loss plateaus or begins to rise, it signals that overfitting has begun.
3.  **Action:** Halt training and save the model parameters from the epoch where validation performance was at its peak.

### Synthesis: The Holistic View
A robust modern deep learning workflow often combines these techniques:
*   Use **L2 Regularization** (or AdamW) for stability.
*   Apply **Dropout** to prevent co-adaptation.
*   Use **BatchNorm/LayerNorm** to stabilize feature distributions.
*   Implement **Early Stopping** using the validation set to determine the optimal stopping point.