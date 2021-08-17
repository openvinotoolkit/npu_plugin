# Software Layers HLD

This document describes the proposed HLD for generic software layers support in **VPUX NN compiler**.
The design is supposed to be used for MTL/LNL activation SHAVE kernels, but can also be extended
to KMB UPA SHAVEs if needed.

## Terminology

* **Kernel** - the entity, which holds the machine code for the kernel and its RO data.
  In general, it is an ELF file in memory, with `.text` section.
  Run-time will run the entry point function from `.text` section during inference.
* **Kernel Invocation** - particular call of the **Kernel** with provided arguments and optional RW data.
* **Kernel Task** - entity for scheduling by the runtime. Includes **Kernel Invocation**,
  barriers configuration and tile selection.
* **Runtime Kernel** - special management **Kernel**, which performs scheduling of the **Kernel Task**.
  In particular, it takes the tasks from FIFO, waits its barriers, launch its with provided arguments,
  updates producer barriers and perform other management work.
  This kernel is also provided by the **VPUX NN compiler** and is stored in the VPU blob.

## Memory Configuration

The SW kernels uses windows-based virtual addressing, which means that all pointers inside the **Kernel**
belongs to the one of the following window:

* `WIN_D` - **Kernel** code + RO data
* `WIN_E` - **Kernel Invocation** arguments + RW data
* `WIN_F` - CMX slice per SHAVE tile, it will hold the input/output buffers.

Thus, compiler can generate relocatable code and arguments list, fully ready for execution.
All addresses will be represented as `window_start_address` (for example, `0x1F000000`) plus relative offset.

In addition, the compiler will provide SHAVE stack configuration for the particular inference.
It will include per-tile stack sizes for each SHAVEs.

## Kernel Arguments Contract

The format of particular **Kernel** arguments must be fixed between compiler and the kernel and represented as a contract between them.
It includes the order of arguments as well as their internal structure for complex cases.

For example, there should be a single definition of the buffer argument:

```C++
struct __attribute__((packed)) MemRefData {
    uint32_t dataAddr;  // Can't use pointers, since they have platform-dependent size.
                        // Will be located in WIN_F.

    uint32_t isStatic;  // Boolean flag to indicate static shape vs dynamic shape.

    uint32_t numDims;
    uint32_t dimsAddr;      // Pointer to the buffer with dimensions (int32_t[]).
    uint32_t stridesAddr;   // Pointer to the buffer with strides in bits (int64_t[]).
                            // Will be located in WIN_E (static case) or in WIN_F (dynamic case).
                            // The kernel should infer output dims/strides and write them only in dynamic case.

    uint32_t dataType;      // An enum, which should be aligned between kernels and the compiler.
    uint64_t dimsOrder;     // Packed permutation array.
};
```

The compiler will create those structures for all buffer arguments of the **Kernel Invocation**.

Each particular **Kernel** will have it's own parameters structure corresponding to original NN layer:

```C++
struct __attribute__((packed)) ClampParams {
    MemRefData input;
    MemRefData output;
    f16 minVal;
    f16 maxVal;
};
```

With this fixed definition of the parameters, the **Kernel** itself can be implemented as C++ function,
which takes raw pointer to the arguments list and casts it to its parameter structure.
The compiler will guarantee the match of the provided arguments buffer with the contract.

```C++
void clamp(uint32_t argsAddr, uint32_t argsSize) {
    assert(argsSize == sizeof(ClampParams)); // Just for debugging, should be removed from Release code.

    ClampParams* params = reinterpret_cast<ClampParams*>(static_cast<uintptr_t>(argsAddr));

    for (...) {
        ...
    }
}
```

## Kernel Representation in the Compiler

The SW kernel will be represented in the Compiler in several parts:

1. Function declaration, which holds the arguments contract between compiler and the **Kernel**.
2. The information about location of the **Kernel** code or the implementation of the auto-generated **Kernel** function.
3. `VPUIP.SW.Kernel` operation, which will hold the reference to the **Kernel** function and which will provide common operation interfaces.
4. `VPUIP.SW.Kernel.run` operation, which will be used in the `VPUIP.SW.Kernel` inner region and which will perform arguments mapping.
5. `VPURT.Task` operation, which will hold common scheduling information (barriers and executor).

Summarizing the above, the SW kernel will be represented in the following way at the compiler final stages:

```MLIR
// Top-parent module, which holds the entire network/blob.
module @network {

// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func private @builtin_clamp(%input : memref<*xf16,>, %output : memref<*xf16>, %minVal : f16, %maxVal : f16)
        attributes { VPU.kernel_code = <place where to get the code> }
}

// The network main entry point, which will be translated to MappedInference or it's analogue.
func @main(...) {
    // Static barriers declarations.
    %b0 = VPURT.ConfigurePhysicalBarrier <0>
    %b1 = VPURT.ConfigurePhysicalBarrier <1>

    // Task list for DMA engine.
    VPURT.TaskList @DMA {
        // The DMA task, which copies the input tile from DDR to CMX prior to SW kernel invocation.
        VPURT.Task updates(%b0) {
            VPUIP.DMA inputs(%in_tile_0_ddr) outputs(%in_tile_0_cmx)
        }

        // The DMA task, which copies the output tile from CMX to back to DDR after the SW kernel invocation.
        VPURT.Task waits(%b1) {
            VPUIP.DMA inputs(%out_tile_0_cmx) outputs(%out_tile_0_ddr)
        }
    }

    // Task list for Activation SHAVEs.
    VPURT.TaskList @ACT_SHAVE {
        // Particular Kernel invocation.
        VPURT.Task waits(%b0) updates(%b1) {
            // Genetic Kernel information for the scheduler.
            VPUIP.SW.Kernel
                    @VPU.SW.builtin_clamp               // The reference to the Kernel function.
                    on tile 0                           // The tile index to execute on.
                    inputs(%in_tile_cmx_0 as %arg0)     // Inputs/outputs buffers for generic operation interface
                    outputs(%out_tile_cmx_0 as %arg1)   // and their mapping to inner region.
            {
                // Inner region, isolated from above, which holds the information about arguments mapping.

                // We can use constant scalars/arrays definitions here.
                %minVal = constant 0.0 : f16
                %maxVal = constant 6.0 : f16

                // The arguments mapping, the order must match the kernel parameter structure.
                VPUIP.SW.Kernel.run(%arg0, %arg1, %minVal, %maxVal)
            }
        }
    }
}

}
```

The **Runtime Kernel** should also be represented as function declaration with information about the machine code location.
It should be referenced by `VPURT.Graph` meta-operation, which also should store the SHAVE stacks configuration.

```MLIR
module @network {

module @VPU.SW {
    func private @runtime()
        attributes { VPU.kernel_code = <place where to get the code> }
}

VPURT.Graph {
    act_shaves : {
        runtime: @VPU.SW.runtime,
        stack_configuration: [
            128000, // Size in bytes for the SHAVEs in the first tile.
            128000  // Size in bytes for the SHAVEs in the second tile.
        ]
    }
}

}
```

## Kernel Serialization

**Note:** This part is bound to current FlatBuffer-based VPU blob format.
It should be adapted for other formats like ELF.

The VPU Blob uses the following structures to hold the information about the SW kernels:

* `KernelData` - raw binary buffer, which holds **Kernel** ELF file and **Kernel Invocation** arguments buffer.
* `KernelDataReference` - sub-view inside `KernelData`, which refers to particular part (ELF `.text` section, for example).
* `ActKernel` - structure, which holds **Kernel** information:
  * `KernelDataReference` to `.text` section.
  * Entry point offset inside the `.text` section. **TODO:** should be always assume `0` here?
* `ActKernelInvocation` - structure, which holds **Kernel Invocation** information:
  * Reference to the corresponding `ActKernel`.
  * `KernelDataReference` to the arguments buffer.
  * `KernelDataReference` to the SW `.data` section.
  * Tile index to launch the kernel on.
* `ActKernelTask` - holds `ActKernelInvocation` and scheduling related information (like barriers configuration).
* `ActKernelRuntime` - holds the information about **Runtime Kernel** and generic run-time configuration:
  * Reference to the corresponding `ActKernel`.
  * SHAVE stack configuration.

The serialization should be implemented in the following way:

1. The backend should enumerate all functions inside inner `@VPU.SW` module and get their final machine code (`.text` section).
   It can take them from file or compile on-the-fly. The backend should serialize those code into `KernelData` structure and
   add to the common `kernelData` section inside VPU blob. It should also create single `ActKernel` descriptor per each function and
   store it in internal map for future access.
2. For each `VPUIP.SW.Kernel` in the main entry point function body the backend should take serialized `ActKernel` from its inner map,
   pack the parameters taken from inner `VPUIP.SW.Kernel.run` operation into single buffer (with windows-based virtual addressing) and
   create `ActKernelInvocation` structure from them. On top of this structure the generic `ActKernelTask` will be created with corresponding
   barriers configuration description.

## Issues with Current DPU Tasks Representation

In the current scheme the DPU tasks are represented as inner parts of single NCE task.
This contradicts with run-time logic, since run-time uses NCE task variant as actual entity of scheduling.

Also in contrast to SW kernels, which uses per-invocation barrier configuration,
DPU tasks use shared barriers configuration stored in common invariant part.

This might introduces stalls between DPUs and ACT SHAVEs in MTL/LNL cases.

For example, for the tensor in the CMX memory (either the full tensor or a slice of larger buffer from DDR)
the compiler will generate per-DPU split of the workloads.
After DPU finishes one workload, the ACT SHAVE can immediately start post-processing kernel on it.
But it is not possible right now, since ACT SHAVE Kernel Task have to wait common barrier for all DPUs,
so it won't start until all workloads are finished.

## Changes for Schema and Run-time

* Merge common arguments into per-invocation arguments, it's up to compiler to optimize its memory footprint.
* Make an argument list just a "black box" binary buffer from run-time perspective.
* Provide just a per-tile stack size without content. No need for per-SHAVE configuration,
  since compiler is not able to select particular SHAVE for execution, only tile.
* Use single invocation per task - this is closer to run-time scheme of scheduling.
* Use per-DPU task (per-variant) barriers configuration to match SW kernels behavior and avoid stalls between them.
