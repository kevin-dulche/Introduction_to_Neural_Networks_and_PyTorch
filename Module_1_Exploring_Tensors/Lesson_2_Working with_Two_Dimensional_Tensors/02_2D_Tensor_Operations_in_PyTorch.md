# Common Operations in 2D Tensors Using PyTorch

---

### Module Objective

By the end of this module, you will be able to:
*   Perform advanced indexing and slicing operations on structured 2D tensors.
*   Apply fundamental arithmetic operations (addition, scaling) while adhering to size requirements and broadcasting rules.
*   Differentiate between element-wise multiplication (Hadamard) and matrix multiplication ($\mathbf{A} \cdot \mathbf{B}$).
*   Understand how tensor shape and rank govern the validity of all advanced operations.
*   Recognize how 2D concepts extend to higher dimensions (3D, 4D) for complex data like color images.

***

## I. Data Access: Indexing and Slicing in 2D Tensors

Since a 2D tensor is structured like a grid (Rows $\times$ Columns), its access methods are crucial:

*   **Indexing:** Accesses a single specific element using two indices: `Tensor[row, column]`.
    *   *Example:* Retrieving the value at Row 0, Column 1 (`A[0, 1]`).
*   **Slicing:** Retrieves structured sections of data.
    *   `A[i, :]`: Selects an entire row (all columns for a given row index).
    *   `A[:, j]`: Selects an entire column (all rows for a given column index).
    *   `A[:, start:end]`: Selects a vertical strip of data (a range of columns across all rows).

> **ML Context:** This precision is vital for feature selection and debugging, allowing engineers to isolate the impact of specific features or samples.

## II. Arithmetic Operations on 2D Tensors

### A. Addition (Element-wise)
*   **Operation:** Element by element addition ($\mathbf{A} + \mathbf{B}$).
*   **Requirement:** Both tensors must have identical sizes, unless broadcasting rules apply.
*   **ML Use Case:** Combining different sets of feature inputs or accumulating partial gradient contributions.

### B. Scalar Multiplication
*   **Operation:** Multiplies every single element in the tensor by a constant scalar ($c \cdot \mathbf{A}$).
*   **Use Case:** Scaling features (e.g., normalizing all weight parameters) or adjusting gradients by a learning rate ($\eta$).

### C. Hadamard Product (Element-wise Multiplication)
*   **Operation:** Multiplies corresponding elements of two tensors ($\mathbf{A} * \mathbf{B}$). **Note:** This is NOT matrix multiplication.
*   **Requirement:** Tensors must be broadcast compatible.
*   **ML Use Case:** Used in mechanisms like attention, where feature interactions are weighted element-by-element.

### Matrix Multiplication (Dot Product)
This operation is the bedrock of linear algebra in deep learning and has distinct rules:
*   **Operation:** Follows standard linear algebra rules, involving the dot product between rows of $\mathbf{A}$ and columns of $\mathbf{B}$. ($\text{Output}_{ij} = \sum_k A_{ik} B_{kj}$).
*   **Requirement (Crucial):** The number of **columns in the first matrix ($\mathbf{A}$) MUST equal the number of rows in the second matrix ($\mathbf{B}$)**.
*   **ML Use Case:** This is how inputs are multiplied by weight matrices in every single neural network layer—the forward pass computation.

## III. Dimension Management and Extension

### Rank vs. Shape
*   **Shape:** The explicit dimensions (e.g., `(3, 4)`).
*   **Rank:** The number of dimensions/axes (`ndim` = 2 for a 2D tensor).
*   **Importance:** Correctly checking the shape and rank is the single most common source of runtime errors in deep learning pipelines.

### Extending to Higher Dimensions (3D+)
The concepts learned naturally extend:
*   **3D Tensor:** Adds an extra dimension (e.g., Height $\times$ Width $\times$ Channels). The classic example is a color image, where the third axis holds Red, Green, and Blue channels ($\text{Tensor}[H, W, C]$).
*   **Application:** Operations like addition or scalar multiplication extend naturally to 3D tensors, which are essential for Convolutional Neural Networks (CNNs) and video processing.

***
By mastering these operations, you move beyond simple data representation and gain the ability to execute complex mathematical transformations that define modern deep learning algorithms.