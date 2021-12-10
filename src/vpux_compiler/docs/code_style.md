# Style guide
This guide is not intended to cover all possible coding problems. The ultimate goal of this guideline is:
- formalizing of existing practices
- collection of useful programming techniques aimed at increasing the readability and maintainability of our code base

## Naming

### Variable names

The names of variables (including function parameters) and data members are all lowerCamelCase. 
Data members of classes (but not structs) additionally have initial underscores. 
For instance: **localVariable**, **structDataMember**, **_classDataMember**.

#### Common variable names

```cpp
SmallVector<mlir::Value> newResults;  // OK - lowerCamelCase
SmallVector<mlir::Value> new_results; // BAD - lowercase with underscore
```

#### Class data members

Data members of classes, both static and non-static, are named like ordinary nonmember variables, but with an initial underscore.
```cpp
class ClassName final {

...

private:
    Logger _log;      // OK - underscore at the beginning
    static int _cont; // OK
};
```

#### Struct data names

Data members of structs, both static and non-static, are named like ordinary nonmember variables.
```cpp
struct StructName final {
    SmallString entryName;
    SmallString sourceFileName;
    
    static int cont;
};
```

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

## Formatting

### Namespaces

Do not use **llvm::**(or other) namespace prefix for classes imported into **vpux** namespace. 
You can find some of these classes in the [utils](../../../src/vpux_utils/include/vpux/utils/core).

For example using of [SmallVector](../../../src/vpux_utils/include/vpux/utils/core/small_vector.hpp):
```cpp
#include "vpux/utils/core/small_vector.hpp"

void foo(ArrayRef<int> array) {
    SmallVector<mlir::Value> oldResults;       // OK
    llvm::SmallVector<mlir::Value> newResults; // BAD - no need to use llvm:: namespace
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

### Using AliasesInfo

[AliasesInfo](../include/vpux/compiler/core/aliases_info.hpp) is a util class that provides information about the source variable for each alias.
You can get instance of **AliasesInfo** using **getAnalysis** method in function pass:  

```cpp
void OptimizeCopiesPass::safeRunOnFunc() {
    auto& aliasInfo = getAnalysis<AliasesInfo>();
    ...
}
```
  
#### Rules

##### Keep topological order
You have to maintain the topological order of the network throughout the entire time the class is used. 
Otherwise, you may get operations not included in the analysis or the analysis will become incorrect for the new network. 
If you need to change the topological order, you need to be sure that the subgraphs you are processing do not intersect.

These tips may help:  
- Avoid using several patterns with in one pass. This makes it easier to control changes in the topology.
- Prefer using **RetT Operation::walk(FnT &&callback)** than **mlir::OpRewritePattern<OpType>**

##### AliasesInfo and Greedy pattern rewriter
Do not use AliasesInfo in patterns that are controlled with GreedyPatternRewriteDriver. 
GreedyPattern recreates constants and therefore they will not be covered by the aliases analysis.

Bad:
```cpp
void OptimizeCopiesPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto& aliasInfo = getAnalysis<AliasesInfo>();
    
    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<CopyToBlockArgument>(&ctx, aliasInfo, _log);
    
    auto func = getFunction();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}
```

Good:
```cpp
void OptimizeCopiesPass::safeRunOnFunc() {
    auto func = getFunction();
    auto& aliasInfo = getAnalysis<AliasesInfo>();
    func->walk([&](IERT::CopyOp op) {
        fuseLastCopy(op, aliasInfo, _log);
    });
}
```