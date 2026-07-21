# Creating and Managing Datasets in PyTorch

---

### Module Objective

By the end of this module, you will be able to:
*   Implement a custom `Dataset` class that inherits from `torch.utils.data.Dataset`.
*   Define and implement the required methods (`__len__`, `__getitem__`) for proper data access.
*   Apply necessary pre-processing steps using PyTorch's **Transform** system.
*   Combine multiple transformations efficiently using `transforms.Compose` to create a robust, end-to-end data pipeline.

***

## I. The Role of Custom Datasets (The Data Organizer)

### Why Custom Datasets?
When building ML models, the data is rarely simply loaded; it must be *managed*. The PyTorch `Dataset` class provides the standardized interface to organize how raw data samples are accessed and retrieved, making the training process scalable and efficient.

**Process:** You wrap your raw data (features $\mathbf{X}$ and targets $\mathbf{Y}$) within a custom class that adheres to the PyTorch standards.

### Implementing the Custom Dataset Class
To create a working dataset, you must define a custom class that inherits from `torch.utils.data.Dataset` and implement at least two methods:

1.  **`__init__(self)` (The Constructor):** Initializes the class by storing all raw data attributes (e.g., $\mathbf{X}$ features tensor, $\mathbf{Y}$ target tensor).
2.  **`__len__(self)`:** This method tells PyTorch *how many* samples are in the dataset. It returns a single integer: `return len(self)`.
3.  **`__getitem__(self, index)`:** This is the core retrieval method. When PyTorch needs sample $N$, it calls this method. The function retrieves and returns the pair of features and targets for that specific index: `return self.X[index], self.Y[index]`.

## II. Data Transformation (The Pre-Processor)

Real-world data often requires cleaning, scaling, or modifying before it can be used by a neural network. PyTorch uses **Transform** classes for this pre-processing step.

### The Concept of Transform
A transform is a callable object that takes raw input data and returns a modified version of that data (e.g., scaled values, normalized pixels). This keeps the dataset definition clean while allowing flexible manipulation.

*   **Application:** Transforms are applied immediately after the sample is retrieved from the `__getitem__` method, ensuring every piece of data fed into the model is consistent and correctly formatted.

### Composing Transformations (The Pipeline)
For complex preprocessing (e.g., "first scale X, then normalize Y"), you combine multiple transforms using `torchvision.transforms.Compose`.

*   **Process:** You list transforms in order of execution. When a sample is retrieved, the data automatically passes through each transform sequentially (`Transform A` $\rightarrow$ `Transform B` $\rightarrow$ Final Sample).
*   **Benefit:** This creates an efficient, readable, and reusable data pre-processing pipeline.

## III. The Complete Workflow: Dataset $\to$ DataLoader

While this module focuses on the `Dataset`, it's vital to know the next step:

1.  **`Dataset` (The Manager):** Knows *how* to find Sample $N$.
2.  **`DataLoader` (The Feeder):** Takes the `Dataset` and handles the heavy lifting of loading, **batching**, shuffling, and parallelizing the data retrieval—ensuring that tensors are fed into the model in optimized chunks for efficient training.