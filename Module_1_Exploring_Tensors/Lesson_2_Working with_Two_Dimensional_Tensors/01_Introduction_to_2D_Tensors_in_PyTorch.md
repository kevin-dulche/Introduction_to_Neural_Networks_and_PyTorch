# Introduction to 2D Tensors in PyTorch

---

### Module Objective

By the end of this module, you will be able to:
*   Define a 2D tensor and understand its structure as a matrix (rows and columns).
*   Utilize key attributes (`.shape`, `.ndim`) to correctly analyze the dimensions of structured data.
*   Apply advanced indexing and slicing techniques to access specific rows, columns, or sub-matrices.
*   Understand why 2D tensors are fundamental for representing structured datasets and images in machine learning.

***

## I. Defining the 2D Tensor Structure

### What is a 2D Tensor?
A 2D tensor is a matrix—an array of numbers arranged systematically in **rows** (horizontal) and **columns** (vertical). It represents data in a structured, grid-like format.

*   **Representation:** $\text{Tensor}[i, j]$ where $i$ is the row index and $j$ is the column index.
*   **ML Use Cases:**
    1.  **Tabular Datasets:** Rows represent individual **samples** (data points); Columns represent the distinct **features** (variables).
    2.  **Images:** Each pixel value is a grid coordinate, making an image fundamentally a 2D tensor (for grayscale images) or a 3D/4D tensor (when considering color channels and batches).

### Key Attributes for Inspection
Understanding these properties is crucial for debugging data flow:

*   `.ndim` or `.dim()`: Returns the number of dimensions (axes) the tensor has (must be 2 in this case).
*   `.shape`: Returns a tuple defining the size along each dimension, typically `(Rows, Columns)` or `(Batch Size, Features)`.
*   `.size()`: Provides the same information as `.shape()`.
*   `.numel()`: Returns the total number of elements in the entire tensor ($R \times C$).

## II. Accessing and Manipulating Data (Indexing & Slicing)

### Indexing
To access a single element, you need two indices: `tensor[row_index, column_index]`.
*   **Purpose:** Direct retrieval of specific data points.

### Slicing (The Power Tool)
Slicing allows you to retrieve entire contiguous sections of the tensor without accessing individual elements.
*   **Rows Only:** Retrieving a single row: `tensor[i, :]` (selects all columns for row $i$).
*   **Columns Only:** Retrieving a single column: `tensor[:, j]` (selects all rows for column $j$).
*   **Sub-matrices:** Retrieving a rectangular block of data using two pairs of indices.

## III. Importance in Machine Learning Workflows

The structured nature of 2D tensors makes them indispensable to ML models because they align perfectly with how we conceptualize real-world data:

1.  **Dataset Representation:** The standard structure is **[Samples, Features]**. Each row is a complete sample; each column represents one measurable characteristic (feature).
2.  **Matrix Algebra Core:** Many foundational ML operations are matrix multiplications, which inherently require 2D tensor inputs. This reinforces the connection between linear algebra and deep learning computation.

---
***Summary: The 2D Tensor organizes structured data into a readable grid format. By mastering its creation, attributes, and powerful slicing mechanisms, you can effectively prepare tabular datasets and images for core machine learning operations like transformation and prediction.***