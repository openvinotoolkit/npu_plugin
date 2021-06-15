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

class createDeclareTensorOp<string memoryLocation> :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::DeclareTensorOp>(
                $_loc, $0.getType(),
                vpux::VPUIP::MemoryLocation::}] # memoryLocation # [{,
                $1.getInt()
            )
        }]>;

class HasMemSpace<string physicalMemory> :
        Constraint<CPred<[{
            $0.getType().cast<mlir::MemRefType>().getMemorySpace() ==
                vpux::VPUIP::PhysicalMemoryAttr::get(
                    $0.getContext(), vpux::VPUIP::PhysicalMemory::}] # physicalMemory # [{
                )
        }]>>;

def RewriteStaticAllocDDR :
        Pat<
            (IERT_StaticAllocOp:$memref $offset),
            (createDeclareTensorOp<"VPU_DDR_Heap"> $memref, $offset),
            [
                (HasMemSpace<"DDR"> $memref)
            ]
        >;

def RewriteStaticAllocNNCMX :
        Pat<
            (IERT_StaticAllocOp:$memref $offset),
            (createDeclareTensorOp<"VPU_CMX_NN"> $memref, $offset),
            [
                (HasMemSpace<"CMX_NN"> $memref)
            ]
        >;

//
// IERT.Constant -> VPUIP.DeclareConstantTensor
//

def createDeclareConstantTensorOp :
        NativeCodeCall<[{
            $_builder.create<vpux::VPUIP::DeclareConstantTensorOp>(
                $_loc, $0.getType().cast<mlir::MemRefType>(), $1, getConstLocaleIndex($0)
            )
        }]>;

def RewriteConstant :
        Pat<
            (IERT_ConstantOp:$output $value),
            (createDeclareConstantTensorOp $output, $value)
        >;

#endif
