# VPUX NN Compiler

## Introduction

The **VPUX NN Compiler** is a new NN compiler for VPU platforms.
It is based on [MLIR framework](https://mlir.llvm.org/) and utilizes its API and features.

### Goals

The **VPUX NN Compiler** is designed to achieve the following goals:

* Improve network coverage for current VPU generation.
* Make new network enablement process lighter and faster.
* Extend compilation features set:
  * Tensors with arbitrary number of dimensions.
  * Dynamic shapes.
  * Control flow.
* Improve compilation performance.
* Integrate existing solutions from different projects in single code base.
* Design future-proof architecture with extensibility to the next VPU generations.
* Improve developer experience (debuggability, self-validation techniques, testing approach).

### MLIR Framework

The **VPUX NN Compiler** utilizes the following feature from MLIR to improve developer experience:

* IR manipulations.
* Transformations and pass management.
* IR self-validation.
* Unit testing.
* Debugging.

More information about the MLIR framework can be found on the [wiki page](https://wiki.ith.intel.com/display/VPUWIKI/MLIR+Framework).

## Design Principles

The **VPUX NN Compiler** architecture and its implementation is based on the following principles:

1. Explicit notion for IR validity and invariants.
2. Enforced architectural stability and self-validation during compilation pipeline.
3. IR splitting onto separate stages with different level of details.
4. Operation interfaces for generic passes.
5. Atomic passes and pipelines.

The first principle is achieved by MLIR architecture - validation hooks.
Each operation/attribute/type has its own validation hook, which checks all invariants of the entity.
Each pass/transformation takes a valid IR as input and produces a valid IR as output.

The second principle is described in details in [separate section](architectural_stability.md).

The third principle is achieved by MLIR architecture - Dialects concept.
The **VPUX NN Compiler** consists of several Dialects with different level of details.
The IR is lowered from high level abstractions to more detailed representation step-by-step during compilation pipeline.

The fourth principle encourages using such MLIR concepts as Operation Traits and Interfaces.
They allow to reduce code duplication and group similar Operations under a single API.
Operation Interfaces also allows to write more generic passes, which are not bound to particular operation set.

The fifth principle declares that each Pass in compilation pipeline must represent one single transformation
to reach one particular goal (either IR adaptation or IR optimization).
Such "atomic" pass is easier to be covered by unit testing.
The "atomic" passes can be joined together in the compilation chain inside pipeline.
The pipeline doesn't perform IR transformation on its own, instead it representes a sequence of other passes.
The goal of pipeline is to establish correct order of underlying passes, while keeping actual transformation logic inside them.

## Topics

* [Architecture](architecture.md)
* [EDSL Integration](edsl.md)
* [Custom Layers](custom_layers.md)
* [Dynamic Shapes Support](dynamism.md)
* [Build and Test Instructions](build_and_test.md)
* [Debugging Technics](debugging.md)
