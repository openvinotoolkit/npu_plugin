//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUREGMAPPED_OPS
#define VPUX_COMPILER_DIALECT_VPUREGMAPPED_OPS

include "vpux/compiler/dialect/VPURegMapped/dialect.td"
include "vpux/compiler/dialect/VPURegMapped/ops_interfaces.td"
include "vpux/compiler/dialect/VPURegMapped/attributes.td"
include "vpux/compiler/dialect/VPURegMapped/types.td"

include "mlir/Interfaces/SideEffectInterfaces.td"

//
// Base classes
//
class VPURegMapped_Op<string mnemonic, list<Trait> traits = []> :
        Op<
            VPURegMapped_Dialect,
            mnemonic,
            traits
        >;

def VPURegMapped_RegisterFieldWrapperOp :
        VPURegMapped_Op<"RegisterFiledWrapper",
            []
        > {
    let summary = "Register Field Wrapper Op";

    let arguments = (ins
        VPURegMapped_RegisterFieldAttr:$regFieldAttr
    );

    let assemblyFormat = [{
        attr-dict
        `regFieldAttr` `(` $regFieldAttr `)`
    }];
}

def VPURegMapped_RegisterWrapperOp :
        VPURegMapped_Op<"RegisterWrapper",
            []
        > {
    let summary = "Register Wrapper Op";

    let arguments = (ins
        VPURegMapped_RegisterAttr:$regAttr
    );

    let assemblyFormat = [{
        attr-dict
        `regAttr` `(` $regAttr `)`
    }];
}

def VPURegMapped_RegisterMappedWrapperOp :
        VPURegMapped_Op<"RegisterMappedWrapper",
            []
        > {
    let summary = "RegisterMapped Wrapper Op";

    let arguments = (ins
        VPURegMapped_RegisterMappedAttr:$regMapped
    );

    let assemblyFormat = [{
        attr-dict
        `regMapped` `(` $regMapped `)`
    }];
}


//
// FetchTaskOp
//

def VPURegMapped_FetchTaskOp:
        VPURegMapped_Op<"FetchTask", [
            DeclareOpInterfaceMethods<VPURegMapped_TaskOpInterface, ["setTaskLocation", "getTaskLocation"]>,
            AttrSizedOperandSegments
        ]> {

    let summary = "Op to represent fetching of specific tasks";
    let description = [{
        FetchTaskOp is indicate copying task descriptor from DDR(blob) to CMX(meta section).
        During IR transformations should be replaced to NNDMA task.
    }];

    let arguments = (ins
        Optional<VPURegMapped_IndexType>:$previousTask,
        VPURegMapped_IndexType:$primary_start,
        VPURegMapped_IndexType:$primary_end,
        Optional<VPURegMapped_IndexType>:$secondary_start,
        Optional<VPURegMapped_IndexType>:$secondary_end
    );

    let results = (outs
        VPURegMapped_IndexType:$index
    );

    let assemblyFormat = [{
        (`previousTask` `(` $previousTask^ `:` type($previousTask) `)` )?
        `primary` `(` $primary_start `->` $primary_end `)`
        `secondary` `(` $secondary_start `->` $secondary_end `)`
        `(` type($primary_start) `->` type($primary_end) `:` type($secondary_start) `->` type($secondary_end) `)`
        `->` type($index)
        attr-dict
    }];
}

//
// EnqueueOp
//

def VPURegMapped_EnqueueOp:
        VPURegMapped_Op<"Enqueue", [
        ]> {

        let summary = "Op to represent actual enquement of a range of contigous tasks";

        let description = [{
            This operation is responsible for the transfer WorkItems from CMS to execution FIFO
        }];

        let arguments = (ins
            Optional<VPURegMapped_IndexType>:$previousTaskIdx,
            VPURegMapped_IndexType:$barrier,
            VPURegMapped_TaskTypeAttr:$taskType,
            VPURegMapped_IndexType:$start,
            VPURegMapped_IndexType:$end
        );

        let results = (outs
            VPURegMapped_IndexType:$index
        );

        let assemblyFormat = [{
            (`previousTaskIdx` `(` $previousTaskIdx^ `:` type($previousTaskIdx) `)` )?
            `at` `(` $barrier `:` type($barrier) `)`
            `(` $start `->` $end `:` type($start) `->` type($end) `)`
            `->` type(results)

            attr-dict
        }];
    }

//
// ExecutionGroupOp
//

def VPURegMapped_ExecutionGroupOp :
    VPURegMapped_Op<"ExecutionGroup", [
        AttrSizedOperandSegments,
        AttrSizedResultSegments
    ]> {

    let summary = "Op to represent a grouping of executable ops that are fetched at once";

    let arguments = (ins
        Variadic<VPURegMapped_IndexType>:$previousTaskIdx,
        Variadic<VPURegMapped_IndexType>:$waitBarriers,
        Variadic<VPURegMapped_IndexType>:$updateBarriers,
        VPURegMapped_TaskTypeAttr:$task_type
    );

    let results = (outs
        Variadic<VPURegMapped_IndexType>:$startIndexes,
        Variadic<VPURegMapped_IndexType>:$endIndexes
    );

    let regions = (region
        SizedRegion<1>:$tasks
    );

    let hasVerifier =1;
}

//
// GroupYieldOp
//

def VPURegMapped_GroupYieldOp :
    VPURegMapped_Op<"GroupYield", [
        HasParent<"ExecutionGroupOp">,
        Terminator,
        AttrSizedOperandSegments
    ]> {

    let summary = "Not physical operation which needed to unite ops in groups which potentially can be fetched together";

    let arguments = (ins
        Variadic<VPURegMapped_IndexType>:$listHeads,
        Variadic<VPURegMapped_IndexType>:$listTails
    );

    let hasVerifier = 1;
}

//
// DeclareTaskBufferOp
//

def DeclareTaskBufferOp :
        VPURegMapped_Op<"DeclareTaskBuffer", [
            NoMemoryEffect
        ]> {
    let summary = "Buffer Op representing a discrete location of a task in runtime metadataBuffer";

    let arguments = (ins
        VPURegMapped_TaskTypeAttr:$task_type
    );

    let results = (outs
        VPURegMapped_IndexType:$index
    );

    let assemblyFormat = [{
        attr-dict
        $task_type `->` type(results)
    }];
}

//
// ViewTaskRangeOp
//

def ViewTaskRangeOp : VPURegMapped_Op<"ViewTaskRange", [
                        NoMemoryEffect
                    ]> {
    let summary = "Task range op, that represents a range of OP's as a memory (memref)";

    let arguments = (ins
        VPURegMapped_IndexType:$first,
        VPURegMapped_IndexType:$last
    );

    let results = (outs
        MemRefRankOf<[UI8], [2]>:$memview
    );

    let assemblyFormat = [{
        `(` $first `->` $last `:` type($first) `->` type($last) `)` `->` type($memview)
        attr-dict
    }];
}

#endif
