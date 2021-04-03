# Architectural Stability of NN Compiler

## Introduction

The stability and debuggability of NN compiler architecture is quite important feature.
It improves developer experience and allow to satisfy the following scalability requirements in faster way:

* New networks support.
* Current supported networks improvements (accuracy and/or performance).
* New compiler features (might be imposed by previous items).

Since the NN compiler contains a set of complex optimization passes with dependencies on each other,
any change in it might introduce new issues.
This increases the time for new features/network enablement, sometimes dramatically.
This document discusses some ways to reduce this complexity on architectural level and speed up development process.

Each NN compiler has its own intermediate representation of the model, which it works on.
The compiler passes modify this model in place and share some information threw it.
The IR itself, just like any other object, should have so called valid state, or some combination of internal invariants.
Each compiler pass shouldn't break this valid state.
The violation of this rule is one of the most cause of issues with new networks/features.

The compiler architecture stability in this point of view is an API property,
which helps developers to keep the IR valid after any transformations done in pass.
This also implies that the API should help developers to focus on actual adaptation/optimization task
rather than on IR validity checking/keeping.
The API should do this automatically and prevent IR transformations that break the combination of internal invariants.

The debuggability in that scope is a set of extra helper methods, which helps to find the IR validity violation.
It should point to exact place, where the issue happened and provide as much information as possible to the developer.
This will reduce time spent on issue debugging.

## IR Valid State

The IR valid state is a combination of different parts:

1. Graph structure validity.
2. Internal information validity.
3. Run-time structure validity.

Each of the parts are covered below in details.

### Graph Structure Validity

The IR used by NN compiler is represented as some graph structure and this implies some graph structure invariants.
This includes:

* Coherency of links between graph objects.
  For example, for node edge `[src] -> [dst]`, the node `[src]` has a link to node `[dst]` and, vice versa,
  the node `[dst]` has a link to node `[src]`. In more complex case the edge itself might be represented as separate object,
  and nodes will have links to that edge object and the edge object will have links to the nodes.
* No cycles in the graph.
* Graph connectedness. There network inputs should be connected with network outputs, in other words,
  we can reach network inputs from network outputs following backward edges.

### Internal Information Validity

The NN compiler IR includes not only the connections between internal elements, but some additional information bound to them.
In most cases, this information is called *attributes* and are bound to graph nodes and, sometimes, to graph edges.
We will use term IR information for the full set of the attributes, bound to its elements.

First, we should divide these attributes onto several sections:

1. **Primary attributes**. Those attributes define the IR, or, in other words, the IR object can't exist without those attributes.
   Example of such attributes: the type of the IR node (data or operation), shape for tensor,
   mandatory parameter for NN layer (strides for Convolution, axis for SoftMax).
2. **Compilation attributes**. Those attributes do not exist at the beginning of the compilation process,
   but are computed and added to the IR during the compilation/adaptation/optimization.
   For example, memory allocation for tensors.
   These attributes will be added after corresponding compiler pass or set of passes are executed.
3. **Computable attributes**. Those attributes can be computed from other attributes.
   For example, full size of tensor can be computed from its shape.

Validity state for those attributes includes their coherency with each other.
For example, axis parameter for SoftMax shouldn't exceed the number of dimensions of the operation input tensor.
But there are also special rules for different attributes kinds.

Since **primary attributes** defines the IR object, there shouldn't be a way to create the object without those attributes.

Since **compilation attributes** arise at some specific point of compilation, before that point the IR shouldn’t contain them at all,
and after that point the attributes must be kept by further transformation.
For example, once the memory allocation pass is done and some memory information is attached to the data objects,
next passes should respect this information, either keep it or re-run the allocation procedure.

As for **computable attributes**, since they are some kind a caching for calculated values,
they should be always up to date with the attributes used for the calculation.

### Run-time Structure Validity

Since NN compiler adopts the model for execution on specific hardware it must consider all limitations of the hardware.
For VPU specific case, where we might have different ways to execute NN layers with different restrictions.
Some restrictions are coming from the hardware level itself (for NCE module, for example),
some restrictions are coming from implementation (for SHAVE kernels).
There also might be performance implications, when several ways are possible, but some of them has penalties.

The IR run-time structure validity means that all these restrictions are satisfied.

## Proposals

This section will present some architectural proposals to make NN compiler more stable in terms of its evolution.

The generic proposal is that all compiler and IR APIs should perform checks for input parameters and for operation transformation
and immediately report in case of errors. This will allow to find errors in pass just in place, where they happened.
The checks must be included into the Release build too.

### Graph Structure Validity Implementation

First, the IR should avoid complicated representation forms and avoid introducing new entities
if something can be represented with existing model.
This will reduce misunderstanding or misuse of its API in passes. It can consist only of Values and Operations.
The computation and execution flow can be fully defined by the dependencies between Values and Operations
as well as internal order of the Operations.
If the Operation takes the Value as input parameter, the Operation that produces this Value should be executed prior to its consumer.
To make additional explicit dependencies between operations (order them to reuse memory, for example),
IR might use Barriers as separate kind of Values. In that case each Operation in addition to Tensor outputs also produces single Barrier output.
Each Operation, in addition to Tensor inputs, might take extra Barrier inputs.
If we’d like to order Operations, that doesn’t have Value dependencies over Tensors, we just link them via Value dependency of Barriers.
From the graph topology point of view, this connection doesn’t differ from connection via Tensors,
so passes should only care about Value dependencies.

Second, since in NN all layers can be considered as pure functions without side effects,
the compiler might automatically remove operations, which outputs are not used.
In that case, to replace the sub-graph, the pass just needs to create new operations and replace the users of one set of Values
to other set of Values. The compiler then might remove unused old operations and check that the graph is still connected
(there is a path from network inputs to network outputs).

### Internal Information Validity Implementation

First, **computable attributes** shouldn’t be stored as normal attributes.
Instead they should be computed on the fly or, for performance reasons, some separate cache mechanism should be used,
which will be automatically notified about change of other attributes.

Second, generic types like `string` or `any` should be avoided for IR objects, limiting the supported attributes for each Operation kind.
Each Operation (represented as separate class) should explicitly mention supported **primary attributes**
(without ability of their modification) and **computation attributes**.

Third, instead of storing some information as attributes of Operations, this information can be represented as separate Operations.
For example, explicit `Quantize`, `Dequantize`, `FakeQuantize` operations instead of quantization parameters attribute.
The compiler passes can detect patterns like `[Dequantize] -> [Convolution]` and merge them.

### Run-time Structure Validity Implementation

As already mentioned, each Operation type should be represented as separate class.
The class should encapsulate all information about the run-time restrictions for that Operation
and check them each time the Operation is modified (eg, its inputs are replaced).
The passes should use some API to get the information about the restrictions rather than assuming them based on the Operation type.

Each change in Operation input should initiate its outputs parameters inference.
The API, that did this modification, should check if something is different in the outputs and continue parameters inference if needed.
