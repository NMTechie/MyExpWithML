### Questions on Tensors
    ❓ **Question:** Why can't tensors store strings?

🧠 **Answer:** Because tensors are optimized for fixed-size numerical types, not variable-length data like strings.

    ❓ **Question:** Define the analogy of vectors in physics vs ML

🧠 **Answer:** Refer this [Answers](DetailedAnswer/AnalogyOfVectorinML.md)
    
    ❓ **Question:** Does numpy ndarray = tensor

🧠 **Answer:** NumPy is not exactly a tensor library, but it's a foundational numerical array library that inspired how tensors are implemented in ML frameworks.
NumPy provides multi-dimensional arrays (ndarray) and basic linear algebra — so yes, it can handle tensor-like data
But NumPy is not designed for:

- Training neural networks

- GPU acceleration

- Auto-differentiation (needed for backpropagation)

Libraries like PyTorch and TensorFlow offer a more complete tensor abstraction, building on the same core concepts — but with ML-specific tooling.

❓ **Question:** Can I say numpy 2d array or matrix = dataframe in pandas

🧠 **Answer:** You're close — but let's clarify the relationship between a NumPy 2D array, a matrix, and a Pandas DataFrame, because while they are similar, they are not exactly the same.

✅ You can say:

A Pandas DataFrame is built on top of a NumPy 2D array and behaves similarly to a matrix in many ways.

But you shouldn't say:

"NumPy 2D array or matrix = DataFrame"

Because that implies they are equivalent or interchangeable, which is not quite true.

| Concept        | NumPy 2D Array              | NumPy Matrix (deprecated) | Pandas DataFrame                     |
| -------------- | --------------------------- | ------------------------- | ------------------------------------ |
| Data Structure | Homogeneous 2D array        | Homogeneous 2D matrix     | Heterogeneous table (rows + columns) |
| Data Types     | Same type (e.g. all floats) | Same type                 | Can have different types per column  |
| Labels         | No labels (just indices)    | No labels                 | Has row and column labels            |
| Functionality  | Optimized for math          | Matrix math (limited use) | Rich data analysis tools             |
| Library        | NumPy                       | NumPy                     | Pandas                               |

🔁 Conversion is possible:

From NumPy array to DataFrame:
```python
import numpy as np
import pandas as pd

arr = np.array([[1, 2], [3, 4]])
df = pd.DataFrame(arr)
```
From DataFrame to NumPy array:
```python
arr = df.values  # or df.to_numpy()
```
✅ Better ways to say it:

"A DataFrame is like a labeled 2D NumPy array."

"A Pandas DataFrame can be created from a NumPy 2D array."

"They are similar in shape and structure, but a DataFrame is more powerful for labeled data."


❓ **Question:** What is the difference between Feature Construction vs Feature Extraction in Feature Engineering

🧠 **Answer:** 

🧩 Feature Construction

Definition:
Feature Construction means creating new features from the existing raw data — usually using domain knowledge or logical combinations of existing features.

Goal:
To enhance the dataset by adding meaningful, interpretable features that improve model performance.

⚙️ Feature Extraction

Definition:
Feature Extraction means transforming existing features into a new, lower-dimensional representation — usually by applying mathematical or statistical techniques.

Goal:
To reduce dimensionality and capture essential information from raw data, especially high-dimensional data like images, text, or signals.

🎯 In Short

Feature Construction: Add new features → enrich data.

Feature Extraction: Compress features → simplify data.


❓ **Question:** What is the difference between Feature Construction vs Feature Extraction in Feature Engineering

🧠 **Answer:** 

Excellent question — standardization and normalization are both feature scaling techniques used in machine learning to make sure features contribute equally to the model, but they do it in different ways and are used for different reasons.

Let’s break them down clearly:

🔹 1. Standardization (Z-score normalization)

Definition:
Transforms data so that it has: Mean = 0 Standard deviation = 1

Formula: 
$$
x' = x-\mu/\sigma
$$
> where
x = original value
μ = mean of the feature
σ = standard deviation of the feature



Effect:

The data is centered around zero.

It keeps the original shape (distribution) of the data but rescales it.

Works well even if data isn’t bounded or is normally distributed.

Typical use cases:

Linear models (e.g., Linear/Logistic Regression)

PCA (Principal Component Analysis)

SVM, K-Means, etc.

Any algorithm that assumes Gaussian-like distributions or uses distances.

🔹 2. Normalization (Min–Max scaling)

Definition:
Rescales features to a specific range, usually [0, 1] (or sometimes [-1, 1]).

Formula:
$$
x'=x-x_{\min}/x_{\max}-x_{\min}
$$
> where $x_{\min}$  and $x_{\max}$ are the minimum and maximum of that feature.

Effect:

All values fall within the same range.

Sensitive to outliers (a single extreme value can skew the scale).

Typical use cases:

Neural networks (e.g., when using sigmoid or tanh activations)

Algorithms that depend on magnitude or bounded inputs (e.g., gradient descent)

Image processing, where pixel values are scaled to [0,1].

🔸 Quick Comparison Table
| Aspect                    | **Standardization**                  | **Normalization**                     |
| ------------------------- | ------------------------------------ | ------------------------------------- |
| Formula                   | ((x - \mu) / \sigma)                 | ((x - x_{min}) / (x_{max} - x_{min})) |
| Range                     | Unbounded                            | [0, 1] (or [-1, 1])                   |
| Handles outliers well?    | ✅ Better                             | ❌ Sensitive                           |
| Keeps distribution shape? | ✅ Yes (shifts & scales)              | ⚠️ Distorts if outliers exist         |
| Typical algorithms        | SVM, PCA, Linear/Logistic Regression | Neural Nets, kNN, Image data          |

✅ Rule of Thumb

Use standardization when the algorithm assumes or benefits from normally distributed data.

Use normalization when the algorithm is sensitive to the scale of data (especially bounded domains).