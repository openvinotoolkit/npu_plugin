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
