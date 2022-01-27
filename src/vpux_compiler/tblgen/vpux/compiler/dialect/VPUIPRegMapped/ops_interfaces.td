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

#ifndef VPUX_COMPILER_DIALECT_VPUIPREGMAPPED_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_VPUIPREGMAPPED_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// TaskOpInterface
//

def VPUIPRegMapped_TaskOpInterface : OpInterface<"TaskOpInterface"> {
    let description = "Interface for VPUIPRegMapped Task";

    let cppNamespace = "vpux::VPUIPRegMapped";

    let methods = [
        InterfaceMethod<
            "Barriers that will free this task to run",
            "mlir::ValueRange", "waitBarriers", (ins),
            [{
                return $_op.waitBarriers();
            }]
        >,
        InterfaceMethod<
            "Barriers that will free this task to run",
            "mlir::MutableOperandRange", "waitBarriersMutable", (ins),
            [{
                return $_op.waitBarriersMutable();
            }]
        >,

        InterfaceMethod<
            "Barriers that will be at least partially unlocked when this task is complete",
            "mlir::ValueRange", "updateBarriers", (ins),
            [{
                return $_op.updateBarriers();
            }]
        >,
        InterfaceMethod<
            "Barriers that will be at least partially unlocked when this task is complete",
            "mlir::MutableOperandRange", "updateBarriersMutable", (ins),
            [{
                return $_op.updateBarriersMutable();
            }]
        >,

        StaticInterfaceMethod<
            "Get the VPUIPRegMapped TaskType for the Operation",
            "vpux::VPUIPRegMapped::TaskType", "getTaskType", (ins),
            [{
                return ConcreteOp::getTaskType();
            }]
        >,
    ];

    let extraTraitClassDeclaration = [{
        void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects) {
            vpux::VPUIPRegMapped::getTaskEffects($_op, effects);
        }

        // 2022_01_22
        //mlir::SymbolRefAttr getExecutor(uint32_t& numUnits)
        vpux::IndexedSymbolAttr getExecutor(uint32_t& numUnits) {
            //return vpux::VPUIPRegMapped::getTaskOpExecutor($_op, numUnits);
            /*
            // Inspired from src/dialect/VPUIP/ops_interfaces.cpp, method vpux::VPUIP::getExecutorAttr()
            return IndexedSymbolAttr::get(kindAttr);
            */

            // Note: from src/core/attributes/indexed_symbol_attr.cpp, method: IndexedSymbolAttr IndexedSymbolAttr::get(mlir::StringAttr name)
            //auto& ctx = getContext();
            // error: conversion from ‘const char [14]’ to non-scalar type ‘mlir::StringAttr’ requested: mlir::StringAttr strAttr = "Alex Test 123";
            //mlir::StringAttr strAttr = mlir::StringAttr::get(&ctx, "VPUX Compiler");
            mlir::StringAttr strAttr = mlir::StringAttr::get(NULL, "VPUX Compiler");
            return IndexedSymbolAttr::get(strAttr);
        }
    }];
}

//
// Legacy4D
//

def VPUIPRegMapped_Legacy4D : NativeOpTrait<"Legacy4D"> {
    let cppNamespace = "vpux::VPUIPRegMapped";
}

#endif
