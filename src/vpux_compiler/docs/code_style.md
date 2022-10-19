# Style guide
This guide is not intended to cover all possible coding problems. Its ultimate goal is:
- to formalize the existing practices;
- to collect some useful programming techniques aimed to enhance the readability and maintainability of our code base.

## Naming

### Variables and functions

The names of common variables (including function parameters) and free functions are all `lowerCamelCase` aka `camelBack`:

```cpp
int someVariable;  // OK
int SomeVariable;  // BAD: it's UpperCamelCase, we only use it for class and struct names
int some_variable; // BAD: it's lower_snake_case
int somevariable;  // TOO BAD

void someFunction(int functionOperand) {                  // OK
    int localVariable = anotherFunction(functionOperand); // OK
    auto lambdaFunction = [](int lambdaOperand) {         // OK
        // <...>
    };
    // <...>
```

### Structures

Structures' names are `UpperCamelCase` aka `PascalCase`
Their data members (incl. `static` ones) and methods are named like ordinary non-member variables and functions:

```cpp
struct SomeStructure final {            // OK: UpperCamelCase for structure names
    void someMethod(int methodOperand); // OK: lowerCamelCase for structure data members and methods
    int dataMember;
    static int staticDataMember;
};
```

### Classes

Classes themselves are named the same way as structures.
The only difference is that their data members (incl. `static` ones) have **an initial underscore (!)**:

```cpp
class SomeClass final { // OK: UpperCamelCase for class names
public:
    void someMethod(int methodOperand) const; // OK: lowerCamelCase for methods

private:
    int _classDataMember; // OK: underscored _lowerCamelCase for all data members of classes
    static int _staticDataMember;
};
```



### Semantic correctness

Do your best to name the things in a most descriptive way, even when it makes the names longer.
Some common abbreviations are acceptable when you're sure they cannot be misunderstood.
Ideally, the reader who finds a use-case of your object should be able to figure out what it is and why it's there with no need to look for the place where it was defined or initialized.

These tips may help:
- Focus on describing **"what exactly it is?"** when naming a thing.
- Stick to using **nouns and adjectives for objects** but **verbs and adverbs for functions**: `auto goodDream = person.sleepWell()`.
- For Booleans, stick to using **statement = question()** form: `const bool somethingIsDone = isSomethingDone()`.
- Ask yourself if an _unprepared reader will get the point_ without having to delve too deeply into the surrounding context.

```cpp
int sourceStride; // OK, just minor context understanding is required (e.g. read the function name)
int srcStride;    // OK: "src" is quite commonly used for "source", same as "num" for "number", etc.

int sStride;      // BAD: "s" for what?
int intSrcStride; // BAD: do not mention type in the name, it can be easily infered from the code in any modern IDE
int ss, t, a, k;  // TOO BAD

// BAD: what are these things? Need to go see all the three definitions (lucky if they're 'const's at least)
if (temp / len < maxCount) {
    // <...>
}
```

## Methods

### Arguments

#### Base types

Usually the types provided by MLIR framework such as `mlir::Value`, `mlir::Type`, `mlir::Attribute`
are lightweight objects that only contain a pointer to the implementation:

```cpp
protected:
    ImplType *impl;
```

Pass such types **by value**, not by reference
The same rule applies to all operations from MLIR dialects: `IE`, `IERT`, `VPUIP`, etc.

```cpp
void someFunction(mlir::Value functionOperand);  // OK
void someFunction(mlir::Value& functionOperand); // BAD: no performance gain, so pass it by value instead
```

#### Arrays
If you are passing an array to a function that does not modify it, use `mlir::ArrayRef`.

> **Be aware:** This class does not own the underlying data, it is only expected to be used in
situations where the data resides in some other buffer whose lifetime extends past that of
the `ArrayRef`. Thus, it is not generally safe to store an `ArrayRef`.

```cpp
void justPrintArray(ArrayRef<mlir::Value> array);           // OK
void justPrintArray(const SmallVector<mlir::Value>& array); // BAD: pass it by ArrayRef instead
```

## Formatting

### Indentation

Follow [1TBS aka OTBS](https://en.wikipedia.org/wiki/Indentation_style#Variant:_1TBS_(OTBS)) ("one true brace style"). 
In other words, use brackets with all constructions: function definition, `if`, `else`, `while`, etc.
The readability of this principle is debatable. But this allows us to achieve that insertion of a new line of code anywhere is always safe.

```cpp
void someFunction(mlir::Operation* op) { // OK
    auto layer = mlir::dyn_cast<VPUIP::LayerOpInterface>(op);
    if (layer == nullptr) { // OK
        return;
    }

    // <...>
}

void someFunction(mlir::Operation* op)  // BAD: the opening bracket is at the next line but it should be here
{
    auto layer = mlir::dyn_cast<VPUIP::LayerOpInterface>(op);
    if (layer == nullptr) // BAD: even single-line constructions must have brackets
        return;
}

```

### Surplus namespaces

Do not use `llvm::` (or other) namespace prefix for classes imported into `vpux` namespace.
You can find some of these classes in the [utils](../../../src/vpux_utils/include/vpux/utils/core).

```cpp
void someFunction() {
    SmallVector<mlir::Value> oldResults;       // OK
    llvm::SmallVector<mlir::Value> newResults; // BAD: no need to use llvm namespace
}
```

### Null-pointer checking

To check the validity of a pointer, prefer the most explicit notation i.e. `nullptr` comparison.

```cpp
void someFunction(mlir::Operation* op) {
    auto layer = mlir::dyn_cast<VPUIP::LayerOpInterface>(op);
    if (layer == nullptr) { // OK
        // <...>
    }
    if (layer != nullptr) { // OK
        // <...>
    }
    if (!layer) { // BAD: implicit ptr->bool conversion
        // <...>
    }
}
```

### Understandability: code structure and comments

> _...the ratio of time spent reading versus writing is well over 10 to 1. We are constantly reading old code as part of the effort to write new code._
>    _Robert C. Martin, "Clean Code: A Handbook of Agile Software Craftsmanship"_

Listen to your feelings when you read the code. If you feel like _something here might appear to be not instantly obvious for an unprepared reader_, it's a good reason to intervene. Change the code according to the [KISS principle](https://en.wikipedia.org/wiki/KISS_principle), make it self-documenting, use comments to explain the reasons and choices made.

These tips may help:
- If the possible question starts with **"what"** or **"how"**, the best option usually is to **change the code** itself.
  Examples: _what is going on? How it works? What is this thing? How it's supposed to be used?_
  You can simplify the code, factor out a method or function, rename something for proper self-documenting, etc.
  If change is impossible, add a descriptive comment at least.
- If the possible question starts with **"why"**, the best option usually is to add an **explanatory comment**.
  Examples: _why this thing is here but not there? Why it's done this way but not another? Why it's done at all?_
- It's also nice to add header comments for classes, functions and methods.
  Focus on brief answers for possible **"what?"**s and **"why?"**s required to scroll through the code fluently.
  Examples: _what do I expect from this thing? Why it's here but not there?_
  If the reader wants to know **"how?"**, he/she can dive into the implementation.
- You can use Doxygen comments notation if you think it makes your comments more verbose and explicit.

```cpp
/// @brief Splits a Copy operation into C-wise tiles                    // OK: "What does this thing do?"
/// @details Currently, only C-wise tiling is implemented for 4D shapes // OK: "Why it does what it does?"
/// @return Outputs of the smaller Copy operations created              // OK: "What should I expect from it?"
/// @todo Replace this with a re-use of the generic tiling utilities
SmallVector<mlir::Value> CopyOpTiling::createChannelWiseTiles(VPUIP::CopyOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    // <...>

    const auto numTilesRequired =
            divUp(origInputShape[Dims4D::Act::C],
                  VPUIP::DMA_LIMIT.count() / (fullCopySize / origInputShape[Dims4D::Act::C]).count());
    // BAD. The reader's experience would be something like:
    // 1. What's going on here? O_o
    // 2. ...a plenty of time and nerves is spent to answer 1.
    // 3. Why so difficult? Maybe just divide the full size by the limit value?

    /// Split the shape into packs of full channels
    const auto numTilesRequired =
            divUp(origInputShape[Dims4D::Act::C],
                  VPUIP::DMA_LIMIT.count() / (fullCopySize / origInputShape[Dims4D::Act::C]).count());
    // STILL BAD: do not describe weird code in comments, rewrite it!

    /// We cannot _just_ divide the fullCopySize by sizeLimit to get the number of tiles required
    /// @example: let fullCopySize=48MB, sizeLimit=16MB and IFM.C=4, then it would be 48/16=3 tiles, but
    ///           it's obviously impossible to split 4 channels into 3 tiles each of those would fit the limits
    const auto numChannelsOfFullShape = origInputShape[Dims4D::Act::C];
    const auto singleChannelSize = fullCopySize / numChannelsOfFullShape;
    const auto numChannelsPerTile = VPUIP::DMA_LIMIT.count() / singleChannelSize.count();
    const auto numTilesRequired = divUp(numChannelsOfFullShape, numChannelsPerTile);
    // OK: comments for reasoning, self-documenting code (though you can improve it of course)
```

## Patterns

### Return Early Pattern

["Return Early"](https://medium.com/swlh/return-early-pattern-3d18a41bba8) is the way of writing functions or methods
so that the expected result is returned at the end of the function while the rest of the code terminates the execution
(by returning or throwing an exception) if conditions are not met. It can be considered as a manifestation of the
Open-Closed Principle from SOLID [(OCP)](https://en.wikipedia.org/wiki/Open%E2%80%93closed_principle).

```cpp
// OK: it's easy to manage fail-cases, therefore OCP is preserved
mlir::LogicalResult goodExample(VPUIP::SubViewOp origOp) {
    auto producerSubViewOp = origOp.source().getDefiningOp<VPUIP::SubViewOp>();
    if (producerSubViewOp == nullptr) {
        return mlir::failure();
    }

    if (origOp.static_strides().hasValue() || producerSubViewOp.static_strides().hasValue()) {
        return mlir::failure();
    }

    // <...>

    return mlir::success();
}

// BAD: to manage conditions, we have to move lots of code and introduce excess code nesting
mlir::LogicalResult badExample(VPUIP::SubViewOp origOp) {
    auto producerSubViewOp = origOp.source().getDefiningOp<VPUIP::SubViewOp>();
    if (producerSubViewOp != nullptr) {
        if (!origOp.static_strides().hasValue() && !producerSubViewOp.static_strides().hasValue()) {
            // <...>
                return mlir::success();
            // <...>
        }
    }

    return mlir::failure();
}
```

### Using checked casts

[checked_cast](../../vpux_utils/include/vpux/utils/core/checked_cast.hpp) is a helper which encapsulates run-time check
for `static_cast`. It is advisable to use it where applicable in order to avoid some unforeseen consequences during
casts. Otherwise, it maybe quite hard to investigate that class of problems. Consider the following code snippet:

```cpp
    int32_t i32 = -100;
    uint32_t u32 = (uint32_t)i32;
    assert((i32 / 10) == (u32 / 10));  // Assertion `(i32 / 10) == (u32 / 10)' failed.
```

It might be a bit syntetic, but it serves the purpose of illustration for unexpected execution flow quite obviously.
End-user may get unexpected accuracy regression without any notice. The problem would become more obvious with
`checked_cast`:

```cpp
    int32_t i32 = -100;
    uint32_t u32 = checked_cast<uint32_t>(i32); // Can not safely cast -100 from int32_t to uint32_t
    assert((i32 / 10) == (u32 / 10));
```


## Using AliasesInfo

[AliasesInfo](../include/vpux/compiler/core/aliases_info.hpp) is a util class that provides information about the
source variable for each alias. You can get an instance of `AliasesInfo` using `getAnalysis` method in a function pass:

```cpp
void MyFancyPass::safeRunOnFunc() {
    auto& aliasInfo = getAnalysis<AliasesInfo>();
    // <...>
```

### Rules

#### Keep topological order
You have to maintain the topological order of the network throughout the entire time the class is used.
Otherwise, you may get operations not included in the analysis, or the analysis would become incorrect for the new network.
If you need to change the topological order, make sure that the subgraphs you are processing do not intersect.

These tips may help:
- Avoid using several patterns within one pass. This makes it easier to control changes in the topology.
- Prefer using `RetT Operation::walk(FnT &&callback)` rather than `mlir::OpRewritePattern<OpType>`.

##### AliasesInfo and Greedy pattern rewriter
Do not use `AliasesInfo` in the patterns controlled with `GreedyPatternRewriteDriver`.

```cpp
// OK
void OptimizeCopiesPass::safeRunOnFunc() {
    auto func = getFunction();
    auto& aliasInfo = getAnalysis<AliasesInfo>();
    func->walk([&](VPUIP::CopyOp op) {
        fuseLastCopy(op, aliasInfo, _log);
    });
}

// BAD: Greedy Pattern re-creates constants, therefore they will not be covered by the aliases analysis
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
