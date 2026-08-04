# Introduction to Linear Classifiers: From Scores to Probabilities

---

### Module Objective

This module shifts the focus from minimizing continuous error (regression) to solving discrete classification problems. You will learn how linear models establish decision boundaries and use the sigmoid function to convert raw scores into quantifiable probabilities for probabilistic classification.

***

## I. The Classification Problem
In a binary or multi-class problem, we are no longer predicting a continuous number; we are assigning a *discrete label* (e.g., 0 or 1).

### Data Structure
The inputs ($\mathbf{X}$) remain the feature matrix (Samples $\times$ Features), but the labels ($\mathbf{Y}$) are discrete integers representing categories.

## II. The Decision Boundary: Linear Separation
A linear classifier's job is to draw a boundary in the feature space that separates different classes of data points.

### 1. The Score Calculation (Linear Function)
The model still computes a raw score ($Z$) using the same linear function as regression, but now $Z$ represents the **likelihood score**, not an expected value.
$$Z = \mathbf{W} \cdot \mathbf{X} + b$$

*   **Interpretation:** The sign of $Z$ determines which side of the boundary a sample falls:
    *   $Z > 0$: Sample belongs to Class 1 (Positive Side).
    *   $Z \le 0$: Sample belongs to Class 0 (Negative Side).

### 2. The Limitation of Simple Thresholding
Simply using a threshold on $Z$ gives a hard class label, but provides no information about **confidence**. If $Z=0.1$, the model is barely sure; if $Z=10$, it is highly certain. This ambiguity is the weakness of simple linear classifiers.

### Solution: The Sigmoid Function
To convert the raw score ($Z$) into a meaningful probability, we pass $Z$ through the **Sigmoid function**:
$$p = \frac{1}{1 + e^{-Z}}$$

*   **Output Range:** The sigmoid squashes any real number ($-\infty$ to $\infty$) into a probability value between $(0, 1)$.
*   **Interpretation:** The resulting probability $p$ is the model's confidence score that the sample belongs to Class 1.
    *   $Z \to \text{Large Positive} \implies p \to 1$ (High Confidence for Class 1).
    *   $Z \to \text{Large Negative} \implies p \to 0$ (High Confidence for Class 0).

## III. Multi-Class Extension: The Hyperplane
The concept generalizes naturally regardless of the number of dimensions:
*   **2D:** Decision Boundary is a **Line**.
*   **3D:** Decision Boundary is a **Plane**.
*   **Higher D:** Decision Boundary is a **Hyperplane**.

## IV. Training and Cost (The Loss Function)

### Loss Function for Classification
In classification, we use the **Cross-Entropy Loss**. This loss function is designed to penalize models that are highly confident but wrong.

$$\text{Loss} = -\sum_{c=1}^{C} y_c \cdot \log(\hat{p}_c)$$
(Where $y_c$ is the true probability (one-hot encoding) and $\hat{p}_c$ is the predicted probability.)

**Goal:** Training minimizes this loss, pushing the model to assign high probabilities ($\approx 1$) to the correct class.