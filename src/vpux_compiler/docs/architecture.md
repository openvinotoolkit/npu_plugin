# Architecture

The **VPUX NN Compiler** consists of the following parts:

* **Core utilities**.
* **FrontEnd**.
* **Dialects**
* **Compilation pipelines**
* **BackEnd**.

## Core Utilities

The **VPUX NN Compiler** core utilities includes various auxiliary classes and functions to simplify IR interpretation and transformations:

* `src/experimental/vpux_compiler/include/vpux/compiler/utils/`
* `src/experimental/vpux_compiler/include/vpux/compiler/core/ops_interfaces.hpp`
* `src/experimental/vpux_compiler/include/vpux/compiler/core/static_allocation.hpp`
* `src/experimental/vpux_compiler/include/vpux/compiler/core/attributes/const_content.hpp`

One part of the core utilities is tensor shape/stride/layout manipulation API:

* `src/experimental/vpux_compiler/include/vpux/compiler/core/attributes/dim.hpp`
* `src/experimental/vpux_compiler/include/vpux/compiler/core/attributes/dim_values.hpp`
* `src/experimental/vpux_compiler/include/vpux/compiler/core/attributes/dims_order.hpp`
* `src/experimental/vpux_compiler/include/vpux/compiler/core/attributes/shape.hpp`
* `src/experimental/vpux_compiler/include/vpux/compiler/core/attributes/strides.hpp`
* `src/experimental/vpux_compiler/include/vpux/compiler/core/attributes/stride_reqs.hpp`

The later is described in details in [separate section](tensor_descriptor.md)

### Lazy constant folding

The **VPUX NN Compiler** uses lazy constant folding approach to reduce memory footprint for large constant values (like dense tensors).

Constant Operations in all Dialects supports lazy folding for the following transformations:

* Precision conversion.
* Layout conversion.
* Reshape.

The Operations deals with 2 separate Types:

* **Actual Value Type** - final type of the resulting tensor/buffer.
* **Content Type** - storage type for the attirubute data.

The **VPUX NN Compiler** provides separate `ConstContentAttr` class, which allows to do this transformations on the fly
when the data is accessed.

## FrontEnd

**FrontEnd** is used to import external source into MLIR infrastructure.
It supports the following sources:

* InferenceEngine `CNNNetwork` object - imported as **IE Dialect**.
* **TBD:** RunTime graph blob - imported as **VPUIP Dialect**.

The **FrontEnd** can be called separately by `vpux-translate` tool.
This mode is used for **LLVM LIT** based unit testing, for example.

## Dialects

The **VPUX NN Compiler** defines the following own Dialects:

* [IE Dialect](generated/dialect/IE.md)
* [IERT Dialect](generated/dialect/IERT.md)
* [VPUIP Dialect](generated/dialect/VPUIP.md)

The Dialect are not used in isolation, but combined with MLIR and PlaidML dialects as well as with each other.

## Compilation pipelines

The graphical representation of the **VPUX NN Compiler** compilation process is available on [wiki page](https://wiki.ith.intel.com/pages/viewpage.action?pageId=1798466423).

The **VPUX NN Compiler** defines the compilation process as a set of top-level pass pipelines for various scenarios:

* Reference mode.
* **TBD:** Simple HW mode.
* **TBD:** Advanced HW mode.
* **TBD:** Throughput mode.
* **TBD:** Latency mode.

Pipeline for each mode calls the passes for each Dialect as well as conversion passes to lower from one Dialect to another.

The passes are described in separate section:

* [Dialect Conversion Passes](generated/conversion/passes.md)
* [IE Dialect Passes](generated/dialect/IE/passes.md)
* [IERT Dialect Passes](generated/dialect/IERT/passes.md)
* [VPUIP Dialect Passes](generated/dialect/VPUIP/passes.md)

## BackEnd

**BackEnd** is used to export **VPUX NN Compiler** IR into external output.
It supports the following modes:

* **VPUIP Dialect** serialization to runtime blob format.

The **BackEnd** can be called separately by `vpux-translate` tool.
This mode is used for **LLVM LIT** based unit testing.
