//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#ifndef VPUX_COMPILER_DIALECT_VPURT_TYPES
#define VPUX_COMPILER_DIALECT_VPURT_TYPES

include "vpux/compiler/dialect/VPURT/dialect.td"
include "vpux/compiler/core/type_interfaces.td"

//
// Base classes
//

class VPURT_Type<string name, list<Trait> traits = []> :
        TypeDef<
            VPURT_Dialect,
            name,
            traits
        > {
    let mnemonic = name;
}

//
// Barrier
//

def VPURT_Barrier : VPURT_Type<"Barrier"> {
    let summary = "VPUIP Barrier Type";

    let description = "This object represents closely a Barrier in the device";
}

//
// SparseBuffer
//

def VPURT_SparseBuffer :
        VPURT_Type<
            "SparseBuffer",
            [
                DeclareTypeInterfaceMethods<NDTypeInterface>
            ]
        > {
    let summary = "VPURT Sparse Buffer Type";

    let description = "This object represents a set of memory references that compose sparse data";

    let parameters = (ins
        "mlir::MemRefType":$data,
        "mlir::MemRefType":$sparsity_map,
        "mlir::MemRefType":$storage_element_table
    );

    let builders = [
        TypeBuilderWithInferredContext<
            (ins
                "mlir::MemRefType":$data,
                CArg<"mlir::MemRefType", "{}">:$sparsity_map,
                CArg<"mlir::MemRefType", "{}">:$storage_element_table
            ), [{
            return $_get(data.getContext(), data, sparsity_map, storage_element_table);
        }]>
    ];

    let skipDefaultBuilders = 1;

    let extraClassDeclaration = [{
        static mlir::Value getData(mlir::Value val);
        static mlir::Value getSparsityMap(mlir::Value val);
        static mlir::Value getStorageElementTable(mlir::Value val);
        static mlir::Type getDataType(mlir::Value val);
    }];
}

#endif
