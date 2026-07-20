# Introduction to Matrices and Vectors: The Math of Change in ML
## Learning Goals

After reviewing this module, you will be able to:

* Define vectors and matrices not just mathematically, but also conceptually as objects and operators.
* Explain how matrix multiplication constitutes a linear transformation that changes data (rotating, stretching, etc.).
* Describe fundamental vector operations (dot product, outer product) and their geometric meaning.
* Understand the specialized role of dual vectors in optimization and learning algorithms.

## I. The Mathematical Hierarchy: Building Blocks
Before defining operations, we must establish the structure:

| **Structure** | **Dimensions** | **Definition** |	**Example/ML Context**|
|---------------|-----------------|---------------|-----------------------|
| **Scalar** | 0D  | A single numerical value.                                      |	Temperature, learning rate ($\eta$).                                    |
| **Vector** | 1D  | An ordered collection of numbers (an arrow).	                |Input features in a dataset; weight parameters ($w$).                      |
| **Matrix** | 2D  | A rectangular array of numbers.                                |	The transformation function applied to input data; the Jacobian matrix. |
| **Tensor** | >2D | A generalization of vectors and matrices (e.g., image data).	| Input images, feature maps in deep networks.                              |

## II. Vectors: Representing State and Quantity

A vector is a fundamental mathematical object representing:

1. **Point/Direction**: Geometrically, it is an arrow showing both magnitude and direction.
2. **Abstract Quantity**: In ML, it represents a collection of features for a single data point (e.g., [Age, Height, Income]).
3. **Machine Learning Role**: Vectors represent the input **features**, model **weights**, and optimization **gradients**.

## III. Matrices: The Operator of Transformation

A matrix acts as an operator that transforms one vector into another through multiplication.

### Linear Transformations

When a matrix multiplies a vector (**$y = M \cdot x$**), it produces a new vector, **$y$**. This process is called a linear transformation because:

* Preserves Linearity: Straight lines remain straight; proportional relationships are maintained.
* Actions: Matrices can perform various transformations on the space—scaling (changing length), rotation (changing angle), reflection (flipping across an axis), and shearing (skewing).

**ML Significance**: In deep learning, matrices transform raw input data into new representations that simplify complex patterns, making them easier for the model to learn.

## IV. Essential Vector Operations (Algebra & Geometry)

These operations define how vectors interact with each other and scalars:

| Operation | Formula / Concept | Result | Geometric Meaning | ML Use Case |
|-----------|-------------------|--------|-------------------|-------------|
| **Vector Addition** |	**$v_1 + v_2$** | Vector | Combined effect of two movements/displacements. | Combining different feature sets. | 
| **Scalar Multiplication** | $c \cdot \mathbf{x}$ | Vector | Changes magnitude (length) while preserving direction. | Feature scaling normalization, gradient updates. |
| **Dot Product (Inner)** | **$v_1 \cdot v_2$** |​ Scalar | Measures the alignment/similarity between vectors. | Calculating similarity scores; angle computation; projecting data onto a feature axis. |
| **Outer Product** | **$v_1 \otimes v_2$** | Matrix | Represents the interaction between components of two vectors. | Constructing covariance matrices; modeling complex feature interactions (weights). |

## V. Advanced Concepts: Dual Vectors and Geometry

### The Role of Dual Vectors (Covectors)

While a **vector** describes a quantity or direction, a **dual vector** (**$\mathbf{v^*}$**) represents a linear function that measures how much a given vector changes when subjected to a specific measurement.

* **Function**: They map vectors $\longrightarrow$ scalars ($\text{scalar} = \mathbf{v^*}\cdot\mathbf{v}$)
* **ML Significance**: Dual vectors are critical in optimization and backpropagation, as they naturally express how the scalar **Loss Function** changes with respect to the vector inputs.

### Geometric Interpretation

Understanding linear algebra is fundamentally visual:

1. **Vectors**: Represent movement or location.
2. **Matrices**: Are the rules for reshaping space itself (the transformation).
3. **Dot Product**: Measures how much one vector "points" in the direction of another (projection).
4. **ML Goal**: Through these operations, models learn to map high-dimensional, complex spaces into lower-dimensional, simplified representations that highlight critical information (Feature Mapping and Dimensionality Reduction).