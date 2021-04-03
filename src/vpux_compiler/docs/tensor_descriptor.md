# Tensor Descriptor

## Logical vs Physical dimensions

The **VPUX NN Compiler** uses the following terms for the Tensor dimensions:

* **Logical** dimensions.
* **Physical** dimensions (also mentioned as **Memory** dimensions).

Those terms (**logical** and **physical**) are also applied to tensor shape and strides.
For example, the **logical** shape means that the dimensions sizes are assigned to **logical** dimensions.

**Logical** dimensions are abstracted from actual memory buffer layout.
Their order in tensor shape is fixed and matches InferenceEngine, nGraph and MLIR order.
The actual meaning of each **logical** dimension is a property concrete Operation.
For example, Convolution interprets **logical** shape of activations tensor as `[N, C, H, W]`
and **logical** shape of weights tensor as `[O, I, KY, KX]`.

**Physical** dimensions, in contrast, are bound to actual memory layout and ordered from major (most outer) to minor (most inner).
They are used to work with memory buffers in common efficient way.

Both **logical** and **physical** dimensions are represented as separate classes (which internally holds single integer value - dimension index).
The `Dim` class represents **logical** dimension, while `MemDim` represents **physical** dimension.
These classes don't have implicit casting to integer, only explicit getter method for dimension index.
These classes are used as keys to access corresponding shape and strides arrays instead of plain integers.
In the same way, shape has two implementations (`Shape` and `MemSpace`) and strides (`Strides` and `MemStrides`).
The usage of separate classes (while they have common implementation logic) allows to catch all misuse of those two abstractions at compile time.

## Memory Layout

The `DimsOrder` class represents memory layout information.
It holds permutation array (in packed format) from **logical** dimensions to **physical** dimensions.
This class provides API to convert between those two representations in both way.
The class also provides API to work with MLIR class (`AffineMap`), which represents more generic layout description.

## Strides requirements

The final utility class in this section is `StrideReqs`.
It is used to collect various requirements for strides from different places and
to calculate the strides based on this information.
It supports the following requirements:

* `Any` - means that there is no special requirements for particular dimension.
* `Compact` - the stride for this dimension must not introduce gaps between neighbor elements in this dimension.
* `Aligned` - the byte stride for this dimension must be aligned by particular value.
* `Fixed` - the stride for this dimension must be equal to fixed value.
