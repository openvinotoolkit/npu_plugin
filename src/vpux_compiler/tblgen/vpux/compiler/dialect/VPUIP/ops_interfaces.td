//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUIP_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_VPUIP_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// LayerOpInterface
//

def VPUIP_LayerOpInterface : OpInterface<"LayerOpInterface"> {
    let description = "Base interface for VPUIP Layer Operation";

    let cppNamespace = "vpux::VPUIP";

    let methods = [
        InterfaceMethod<
            "Get all layer input memory buffers",
            "mlir::OperandRange", "getInputs", (ins),
            [{}],
            [{
                return vpux::VPUIP::getLayerInputs($_op);
            }]
        >,

        InterfaceMethod<
            "Get all layer output memory buffers",
            "mlir::OperandRange", "getOutputs", (ins),
            [{}],
            [{
                return vpux::VPUIP::getLayerOutputs($_op);
            }]
        >,

        InterfaceMethod<
            "Get all layer memory buffers",
            "llvm::detail::concat_range<mlir::OpOperand, llvm::MutableArrayRef<mlir::OpOperand>, llvm::MutableArrayRef<mlir::OpOperand>>",
            "getOpOperands", (ins),
            [{}],
            [{
                return llvm::concat<mlir::OpOperand>($_op.getInOpOperands(), $_op.getOutOpOperands());
            }]
        >,

        InterfaceMethod<
            "Get all layer input memory buffers",
            "llvm::MutableArrayRef<mlir::OpOperand>", "getInOpOperands", (ins),
            [{}],
            [{
                return vpux::VPUIP::getLayerInOpOperands($_op);
            }]
        >,

        InterfaceMethod<
            "Get all layer output memory buffers",
            "llvm::MutableArrayRef<mlir::OpOperand>", "getOutOpOperands", (ins),
            [{}],
            [{
                return vpux::VPUIP::getLayerOutOpOperands($_op);
            }]
        >
    ];

    let extraTraitClassDeclaration = [{
        mlir::Value getViewSource(ptrdiff_t resultInd = 0) {
            return vpux::VPUIP::getLayerViewSource($_op, resultInd);
        }

        void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects) {
            vpux::VPUIP::getLayerEffects($_op, effects);
        }
    }];

    let verify = [{
        return vpux::VPUIP::verifyLayer($_op);
    }];
}

//
// InferLayerReturnTypes
//

class VPUIP_InferLayerReturnTypes<int numResult> {
    code impl = [{
        static mlir::LogicalResult inferReturnTypes(
                mlir::MLIRContext*,
                llvm::Optional<mlir::Location>,
                mlir::ValueRange operands,
                mlir::DictionaryAttr,
                mlir::RegionRange,
                llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
            return vpux::VPUIP::inferLayerReturnTypes(operands, }] #numResult# [{, inferredReturnTypes);
        }
    }];
}

//
// TaskOpInterface
//

def VPUIP_TaskOpInterface : OpInterface<"TaskOpInterface"> {
    let description = "Interface for VPUIP Task";

    let cppNamespace = "vpux::VPUIP";

    let methods = [
        InterfaceMethod<
            "Serialize the Task to BLOB format",
            "vpux::VPUIP::BlobWriter::SpecificTask",
            "serialize", (ins "vpux::VPUIP::BlobWriter&":$writer)
        >,

        StaticInterfaceMethod<
            "Get the executor for the Operation",
            "vpux::VPU::ExecutorKind", "getExecutorKind", (ins),
            [{
                return ConcreteOp::getExecutorKind();
            }]
        >,
    ];

    let extraTraitClassDeclaration = [{
        vpux::IndexedSymbolAttr getExecutor() {
            return vpux::VPUIP::getTaskOpExecutor($_op);
        }
    }];
}

//
// DMATypeOpInterface
//

def VPUIP_DMATypeOpInterface : OpInterface<"DMATypeOpInterface"> {
    let description = "Interface for DMA type layers";

    let cppNamespace = "vpux::VPUIP";

    let methods = [
        InterfaceMethod<
            "Sets DMA hardware profiling id",
            "void", "setDmaHwpIdAttr", (ins "mlir::IntegerAttr":$dmaIdAttr),
            [{}],
            [{$_op->setAttr($_op.getDmaHwpIdAttrName(), (dmaIdAttr));}]
        >,
        InterfaceMethod<
            "Returns DMA hardware profiling id",
            "mlir::IntegerAttr", "getDmaHwpIdAttr", (ins),
            [{}],
            [{
                return $_op.getDmaHwpIdAttr();
            }]
        >,
        InterfaceMethod<
            "Get DMA port attr",
            "mlir::IntegerAttr", "getPortAttribute", (ins),
            [{}],
            [{
                return $_op.getPortAttr();
            }]
        >,
        InterfaceMethod<
            "Get DMA port value",
            "int64_t", "getPortVal", (ins),
            [{}],
            [{
                return $_op.getPort();
            }]
        >,
        InterfaceMethod<
            "Set DMA port attr",
            "void", "setPortAttribute", (ins "mlir::IntegerAttr":$portAttr),
            [{}],
            [{
                return $_op.setPortAttr(portAttr);
            }]
        >,
        InterfaceMethod<
            "Get DMA channel attr",
            "VPUIP::DmaChannelTypeAttr", "getChannelTypeAttr", (ins),
            [{}],
            [{
                return $_op.getChannelTypeAttr();
            }]
        >,
        InterfaceMethod<
            "Get DMA channel value",
            "::llvm::Optional<VPUIP::DmaChannelType>", "getChannelType", (ins),
            [{}],
            [{
                return $_op.getChannelType();
            }]
        >,
        InterfaceMethod<
            "Set DMA channel attr",
            "void", "setChannelTypeAttr", (ins "VPUIP::DmaChannelTypeAttr":$channelTypeAttr),
            [{}],
            [{
                return $_op.setChannelTypeAttr(channelTypeAttr);
            }]
        >,
        InterfaceMethod<
            "Get DMA input value",
            "mlir::Value", "getInput", (ins),
            [{}],
            [{
                return $_op.getInput();
            }]
        >
    ];
}

//
// UPATask
//

def VPUIP_UPATask : NativeOpTrait<"UPATask"> {
    let cppNamespace = "vpux::VPUIP";
}

//
// Legacy4D
//

def VPUIP_Legacy4D : NativeOpTrait<"Legacy4D"> {
    let cppNamespace = "vpux::VPUIP";
}

//
// AsyncLayerOpInterface
//

def VPUIP_AsyncLayerOpInterface : OpInterface<"AsyncLayerOpInterface"> {
    let description = "Interface for layers that will be executed asynchronously in separate Executor";

    let cppNamespace = "vpux::VPUIP";

    let methods = [
        InterfaceMethod<
            "Get Executor for the asynchronous launch",
            "vpux::IndexedSymbolAttr", "getExecutor", (ins)
        >
    ];
}

//
// SoftwareLayerOpInterface
//

def VPUIP_SoftwareLayerOpInterface : OpInterface<"SoftwareLayerOpInterface"> {
    let description = "Interface for layers that will be executed on SHAVEs. (For VPUX37XX only)";

    let cppNamespace = "vpux::VPUIP";

    let methods = [
        InterfaceMethod<
            "Get information about kernel's arguments, entry point and etc.",
            "vpux::VPUIP::KernelInfo", "getKernelInfo", (ins)
        >
    ];
}

//
// SameShape
//

def VPUIP_SameShape : NativeOpTrait<"SameShape"> {
    let cppNamespace = "vpux::VPUIP";
}

//
// SameElementType
//

def VPUIP_SameElementType : NativeOpTrait<"SameElementType"> {
    let cppNamespace = "vpux::VPUIP";
}

//
// SameDimsOrder
//

def VPUIP_SameDimsOrder : NativeOpTrait<"SameDimsOrder"> {
    let cppNamespace = "vpux::VPUIP";
}

//
// SameInOutDimsOrder
//

def VPUIP_SameInOutDimsOrder : NativeOpTrait<"SameInOutDimsOrder"> {
    let cppNamespace = "vpux::VPUIP";
}

//
// SameElementType
//

def VPUIP_SameOperandsAndResultElementType  : NativeOpTrait<"SameOperandsAndResultElementType "> {
    let cppNamespace = "vpux::VPUIP";
}


#endif
