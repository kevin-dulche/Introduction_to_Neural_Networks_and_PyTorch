# Sigmoid Function and Probability Thresholding

---

### Module Objective

This module details the mathematical bridge between raw linear scores and actionable probability estimates. You will master how the sigmoid function standardizes output, enabling models to communicate their level of confidence in a classification decision.

***

## I. The Core Problem: Continuous vs. Discrete Output
Linear classifiers first calculate a continuous score ($Z$), but machine learning needs discrete labels (0 or 1). We must convert this raw score into a probability $[0, 1]$.

### Step 1: Linear Scoring ($\mathbf{Z}$)
The initial output is calculated using the linear function:
$$Z = \mathbf{W} \cdot \mathbf{X} + b$$
*   **Output:** A continuous real number ($-\infty$ to $+\infty$).
*   **Meaning:** $Z$ measures how far along a linear hyperplane the data point falls.

### The Sigmoid Function (The Bridge)
The sigmoid function ($\sigma(z)$) acts as a mathematical squasher, mapping the infinite range of $Z$ into the finite probability space $[0, 1]$.
$$\text{Probability } p = \frac{1}{1 + e^{-Z}}$$

**Key Properties:**
*   **Mapping:** Any real number input ($-\infty$ to $\infty$) is mapped to a probability between 0 and 1.
*   **Confidence:** The distance of the output from $0.5$ directly relates to the model's confidence:
    *   $p \approx 1$: High confidence for Class 1.
    *   $p \approx 0$: High confidence for Class 0.
    *   $p = 0.5$: Low confidence (the sample sits right on the decision boundary).

## II. Decision Making: Thresholding
Once we have a probability $p$, we use a simple threshold to assign a final, discrete class label ($\hat{y}$).

### The Binary Classification Rule
The most common threshold is set at $\mathbf{0.5}$:
*   If $p \ge 0.5$: Predict Class 1 (Positive).
*   If $p < 0.5$: Predict Class 0 (Negative).

> **Significance:** This process allows the model to express its *certainty*. It is not just saying "this belongs here," but "I am X% sure that this belongs here."

## III. Summary: The Classification Pipeline
The flow from raw data input to final prediction label is a three-stage funnel:

$$\mathbf{X} \xrightarrow{\text{Linear Score}} Z \xrightarrow{\text{Sigmoid}} p \xrightarrow{\text{Thresholding}} \hat{\mathbf{y}}$$

This structured approach (scoring $\to$ probability $\to$ label) provides the necessary mathematical rigor and interpretability required for high-stakes classification tasks.