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
