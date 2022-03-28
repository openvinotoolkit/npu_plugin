//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_CONVERSION_REWRITERS_CONVERT_DECLARATIONS_TO_VPUIP
#define VPUX_COMPILER_CONVERSION_REWRITERS_CONVERT_DECLARATIONS_TO_VPUIP

include "vpux/compiler/dialect/IERT/ops.td"
include "vpux/compiler/dialect/VPUIP/ops.td"

include "mlir/IR/OpBase.td"

//
// IERT.StaticAlloc -> VPUIP.DeclareTensor
//

class createDeclareTensorOp<string section> :
        NativeCodeCall<[{
            $_builder.create<vpux::VPURT::DeclareBufferOp>(
                $_loc, $0.getType().cast<mlir::MemRefType>(),
                vpux::VPURT::BufferSection::}] # section # [{,
                $1.getInt()
            )
        }]>;

class HasMemSpace<string physicalMemory> :
        Constraint<CPred<[{
            $0.getType().cast<vpux::NDTypeInterface>().getMemoryKind() == vpux::VPU::MemoryKind::}] # physicalMemory
        >>;

def RewriteStaticAllocDDR :
        Pat<
            (IERT_StaticAllocOp:$memref $offset),
            (createDeclareTensorOp<"DDR"> $memref, $offset),
            [
                (HasMemSpace<"DDR"> $memref)
            ]
        >;

def RewriteStaticAllocNNCMX :
        Pat<
            (IERT_StaticAllocOp:$memref $offset),
            (createDeclareTensorOp<"CMX_NN"> $memref, $offset),
            [
                (HasMemSpace<"CMX_NN"> $memref)
            ]
        >;

#endif
