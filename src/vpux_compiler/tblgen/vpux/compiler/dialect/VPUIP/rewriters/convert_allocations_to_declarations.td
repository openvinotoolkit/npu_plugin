//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_VPUIP_REWRITERS_CONVERT_ALLOCATIONS_TO_DECLARATIONS
#define VPUX_COMPILER_DIALECT_VPUIP_REWRITERS_CONVERT_ALLOCATIONS_TO_DECLARATIONS

include "vpux/compiler/dialect/VPUIP/ops.td"

include "mlir/IR/OpBase.td"

//
// VPUIP.StaticAlloc -> VPUIP.DeclareTensor
//

def createDeclareDDRTensorOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPURT::DeclareBufferOp>(
                $_loc, $0.getType().cast<mlir::MemRefType>(),
                vpux::VPURT::BufferSection::DDR,
                $1.getInt()
            )
        }]>;

def createDeclareCMXTensorOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPURT::DeclareBufferOp>(
                $_loc, $0.getType().cast<mlir::MemRefType>(),
                vpux::VPURT::BufferSection::CMX_NN, $0.getType().cast<vpux::NDTypeInterface>().getMemSpace().getIndex().getValue(),
                $1.getInt()
            )
        }]>;

class HasMemSpace<string physicalMemory> :
        Constraint<CPred<[{
            $0.getType().cast<vpux::NDTypeInterface>().getMemoryKind() == vpux::VPU::MemoryKind::}] # physicalMemory
        >>;

def HasIndex:
        Constraint<CPred<"$0.getType().cast<vpux::NDTypeInterface>().getMemSpace().getIndex().hasValue()">>;

def RewriteStaticAllocDDR :
        Pat<
            (VPUIP_StaticAllocOp:$memref $offset),
            (createDeclareDDRTensorOp $memref, $offset),
            [
                (HasMemSpace<"DDR"> $memref)
            ]
        >;

def RewriteStaticAllocNNCMX :
        Pat<
            (VPUIP_StaticAllocOp:$memref $offset),
            (createDeclareCMXTensorOp $memref, $offset),
            [
                (HasMemSpace<"CMX_NN"> $memref),
                (HasIndex $memref)
            ]
        >;

#endif
