# Style guide
This guide is not intended to cover all possible coding problems. The ultimate goal of this guideline is:
- formalizing of existing practices
- collection of useful programming techniques aimed at increasing the readability and maintainability of our code base

## Methods

### Arguments

#### Base types

Usually the types provided by the MLIR framework such as **mlir::Value**, **mlir::Type**, **mlir::Attribute** are lightweight objects that contain only a pointer to the implementation:

```cpp
protected:
    ImplType *impl;
```

Pass such types by value, not by reference.  

Bad:
```cpp
void foo(mlir::Value& operand) {
    ...
}
```

Good:
```cpp
void foo(mlir::Value operand) {
    ...
}
```

The same rule applies to all operations from IE, IERT, VPUIP, etc. dialects.

#### Array
If you are passing an array to a function that does not modify it, then use **mlir::ArrayRef**. 

> Be aware: This class does not own the underlying data, it is expected to be used in
situations where the data resides in some other buffer, whose lifetime
extends past that of the ArrayRef. For this reason, it is not in general
safe to store an ArrayRef.

Bad:
```cpp
void foo(const mlir::SmallVector<int>& array) {
    // print array
}
```

Good:
```cpp
void foo(ArrayRef<int> array) {
    // print array
}
```

## Patterns

### Return Early Pattern

Return early is the way of writing functions or methods so that the expected result is returned at the end of the function 
and the rest of the code terminates the execution (by returning or throwing an exception) when conditions are not met.  

Bad:
```cpp
mlir::LogicalResult ComposeSubView::matchAndRewrite(IERT::SubViewOp origOp, mlir::PatternRewriter& rewriter) const {
    auto producerSubViewOp = origOp.source().getDefiningOp<IERT::SubViewOp>();
    if (producerSubViewOp != nullptr) {
        if (!origOp.static_strides().hasValue() && !producerSubViewOp.static_strides().hasValue()) {
            ....
            return mlir::success();
        }
    }
    
    return mlir::failure();
}
```

Good:
```cpp
mlir::LogicalResult ComposeSubView::matchAndRewrite(IERT::SubViewOp origOp, mlir::PatternRewriter& rewriter) const {
    auto producerSubViewOp = origOp.source().getDefiningOp<IERT::SubViewOp>();
    if (producerSubViewOp == nullptr) {
        return mlir::failure();
    }

    if (origOp.static_strides().hasValue() || producerSubViewOp.static_strides().hasValue()) {
        return mlir::failure();
    }
    ....
    return mlir::success();
}
```
