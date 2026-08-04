# Bernoulli Distribution and MLE: The Probabilistic Foundation

---

### Module Objective

This reading delves into the deep mathematical foundations of binary classification. You will learn that models fundamentally estimate probabilities using distributions, understand how Maximum Likelihood Estimation (MLE) is used to find those optimal parameters, and recognize the crucial link between log-likelihood and cross-entropy loss.

***

## I. The Bernoulli Distribution: Modeling Binary Outcomes

### Definition
The **Bernoulli distribution** is a fundamental probability distribution used when an experiment has only two possible outcomes (binary): Success/Failure, Heads/Tails, 1/0.
*   It is governed by a single parameter, $\theta$: the probability of success ($P(X=1) = \theta$).
*   The probability of failure is simply $P(X=0) = 1 - \theta$.

### Likelihood Calculation (The Evidence)
If we observe a sequence of independent binary outcomes (e.g., coin flips), the **likelihood** is calculated by multiplying the probabilities of every observed outcome:
$$L(\theta | \text{Data}) = \prod_{i=1}^{N} P(y_i | \theta)$$
*   **Example:** Observing $\text{(Heads, Tails, Heads)}$ with $P(H)=0.2$ yields a likelihood of $0.2 \times 0.8 \times 0.2$.

## II. Maximum Likelihood Estimation (MLE)
The goal is to find the parameter $\theta^*$ that maximizes this calculated likelihood function, making it the probability value that best explains the observed data.

### The Numerical Challenge: Log-Likelihood
As $N$ grows, multiplying many probabilities leads to numbers so small they lose precision ($\approx 0$). We use a mathematical trick to solve this stability issue: **Logarithms**.
*   **The Transformation:** Taking the logarithm converts the product of probabilities into a sum of log-probabilities.
$$\text{Maximizing } L(\theta) \quad \iff \quad \text{Maximizing } \log(L(\theta))$$

*   **Benefit:** The operation changes multiplication ($\prod$) into addition ($\sum$), making calculations numerically stable and simple to implement computationally.

## III. Linking MLE to Loss Functions (The Practical Bridge)
In machine learning, we don't maximize the log-likelihood directly; instead, we minimize the **Negative Log-Likelihood**.

*   **Equivalence:** Minimizing the Negative Log-Likelihood is mathematically equivalent to maximizing the Log-Likelihood.
*   **The Result:** For binary classification, minimizing the negative log-likelihood results in using the **Binary Cross-Entropy Loss**, which is the mathematical foundation for PyTorch's `BCELoss` and its generalization for multi-class problems (Categorical Cross-Entropy).

### Summary Table: Concept $\to$ Math $\to$ ML Tool
| Conceptual Step | Mathematical Concept | Computational Tool | Purpose |
| :--- | :--- | :--- | :--- |
| **Binary Outcome** | Bernoulli Distribution $P(\text{success}) = \theta$ | Sigmoid Function | Transforms raw scores to probabilities $[0, 1]$. |
| **Parameter Search** | Maximum Likelihood Estimation (MLE) | Cross-Entropy Loss $\mathcal{L}$ | Finds the parameters that maximize data likelihood. |
| **Implementation** | Negative Log-Likelihood Minimization | PyTorch's `nn.BCEWithLogitsLoss` | Provides a stable, differentiable loss for training models. |