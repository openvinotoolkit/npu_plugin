//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUREGMAPPED_OPS
#define VPUX_COMPILER_DIALECT_VPUREGMAPPED_OPS

include "vpux/compiler/dialect/VPURegMapped/dialect.td"
include "vpux/compiler/dialect/VPURegMapped/attributes.td"
include "vpux/compiler/dialect/VPURegMapped/types.td"

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
// DeclareTaskBufferOp
//

def DeclareTaskBufferOp : VPURegMapped_Op<"DeclareTaskBuffer", []> {
    let summary = "Buffer Op representing a discrete location of a task in runtime metadataBuffer";

    let arguments = (ins
        VPURegMapped_TaskType:$task_type
    );

    let results = (outs
        VPURegMapped_IndexType:$index
    );

    let assemblyFormat = [{
        attr-dict
        $task_type `->` type(results)
    }];
}

#endif
