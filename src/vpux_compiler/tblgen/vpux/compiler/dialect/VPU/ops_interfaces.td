//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_VPU_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_VPU_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// LayerOpInterface
//

def VPU_LayerOpInterface : OpInterface<"LayerOpInterface"> {
    let description = "Base interface for VPU Layer Operation";

    let cppNamespace = "vpux::VPU";

    let methods = [
        InterfaceMethod<
            "Get all layer input tensors",
            "mlir::OperandRange", "getInputs", (ins),
            [{}],
            [{
                return $_op->getOperands();
            }]
        >,

        InterfaceMethod<
            "Get all layer output tensors",
            "mlir::ResultRange", "getOutputs", (ins),
            [{}],
            [{
                return $_op->getResults();
            }]
        >
    ];

    let verify = [{
        return vpux::VPU::verifyLayer($_op);
    }];
}


//
// SparseOpInterface
//

def VPU_SparseOpInterface : OpInterface<"SparseOpInterface"> {
    let description = "Interface for sparse operations";

    let cppNamespace = "vpux::VPU";

    let methods = [
        InterfaceMethod<
            "Marks the sparsity support for the inputs and outputs of an operation",
            "vpux::VPU::SparsitySupport", "sparsitySupport", (ins)
        >,
    ];
}

//
// NCEOpInterface
//

def VPU_NCEOpInterface : OpInterface<"NCEOpInterface"> {
    let description = "Base interface for VPU NCE Operations";

    let cppNamespace = "vpux::VPU";

    let methods = [
        InterfaceMethod<
            "Check layer strategy compatibility",
            "bool", "checkStrategyCompatibility",
            (ins "VPU::MultiClusterStrategy":$strategy)
        >,

        InterfaceMethod<
            "Get NCE operation kernel size",
            "SmallVector<int64_t>", "getKernelSize", (ins)
        >,

        InterfaceMethod<
            "Get NCE operation kernel strides",
            "SmallVector<int64_t>", "getStrides", (ins)
        >,

        InterfaceMethod<
            "Get NCE operation padding",
            "vpux::VPU::PaddingAttr", "getPad", (ins)
        >,

        InterfaceMethod<
            "Get NCE operation PPE",
            "llvm::Optional<vpux::VPU::PPETaskAttr>", "getPPE", (ins),
            [{
                return $_op.ppe();
            }]
        >,

        InterfaceMethod<
            "Set NCE operation PPE",
            "void", "setPPE", (ins "vpux::VPU::PPETaskAttr":$ppe_task_attr),
            [{
                $_op->setAttr($_op.ppeAttrName(), ppe_task_attr);
            }]
        >,

        InterfaceMethod<
            "Add workload",
            "mlir::Operation*", "addWorkload",
            (ins "mlir::OpBuilder&":$builder, "mlir::Location":$loc,
                 "ShapeRef":$offsets, "ShapeRef":$sizes,
                 "vpux::VPU::PaddingAttr":$pad, "MPEMode":$mpeMode, "mlir::IntegerAttr":$clusterId),
            [{}],
            [{
                return vpux::VPU::details::addWorkload($_op.workloads(), builder, loc, offsets, sizes, pad, mpeMode, clusterId);
            }]
        >,

        InterfaceMethod<
            "Get all workloads",
            "mlir::Region&", "workloads", (ins),
            [{
                return $_op.workloads();
            }]
        >
    ];

    let extraTraitClassDeclaration = [{
        mlir::LogicalResult verifyChannels() {
            return mlir::success(
                vpux::VPU::NCEInvariant::isActTypeSupported(
                    $_op->getOperand(0).getType().template cast<mlir::ShapedType>(),
                    mlir::cast<ConcreteOp>($_op).getInputChannelAlignment()) &&
                vpux::VPU::NCEInvariant::isActTypeSupported(
                    $_op->getResult(0).getType().template cast<mlir::ShapedType>(),
                    mlir::cast<ConcreteOp>($_op).getOutputChannelAlignment()));
        }

        static int64_t getInputChannelAlignmentImpl(vpux::NDTypeInterface inputType) {
            return vpux::VPU::NCEInvariant::getAlignment(inputType.getElementType());
        }
        static int64_t getOutputChannelAlignmentImpl(vpux::NDTypeInterface outputType) {
            return vpux::VPU::NCEInvariant::getAlignment(outputType.getElementType());
        }

        int64_t getInputChannelAlignment() {
            const auto inputType = $_op->getOperand(0).getType().template cast<vpux::NDTypeInterface>();
            return ConcreteOp::getInputChannelAlignmentImpl(inputType);
        }
        int64_t getOutputChannelAlignment() {
            const auto outputType = $_op->getResult(0).getType().template cast<vpux::NDTypeInterface>();
            return ConcreteOp::getOutputChannelAlignmentImpl(outputType);
        }

        bool checkChannelRestrictions(int64_t /*channels*/) {
            return true;
        }
    }];

    let verify = [{
        if (vpux::VPU::details::validatePrecisionForNCE($_op).failed()) {
            return mlir::failure();
        }

        if (vpux::VPU::details::validateWorkloadsRegion($_op->getLoc(), mlir::cast<ConcreteOp>($_op).workloads()).failed()) {
            return mlir::failure();
        }

        return mlir::cast<ConcreteOp>($_op).verifyChannels();
    }];
}

//
// TilingBuilderOpInterface
//

def VPU_TilingBuilderOpInterface : OpInterface<"TilingBuilderOpInterface"> {
    let description = "Interface for operations to reify the provided tiling configuration";

    let cppNamespace = "vpux::VPU";

    let methods = [
        InterfaceMethod<
            "Return the maximum number of tiles for each dimension",
            "SmallVector<int64_t>", "getMaxNumTiles", (ins),
            [{}],
            [{
                return vpux::getMaxNumTiles($_op);
            }]
        >,

        InterfaceMethod<
            "Infer input tiles information from the output tile",
            "vpux::TilingInfo", "backInferTileInfo",
            (ins "const vpux::TileInfo&":$outputTile, "vpux::Logger":$log)
        >,

        InterfaceMethod<
            "Adjust the operation attributes to the current tile properties",
            "void", "adjustAttrs",
            (ins "const vpux::TilingInfo&":$inputTiling, "const TileInfo&":$outputTile)
        >,
    ];
}

//
// TilingInfoOpInterface
//

def VPU_TilingInfoOpInterface : OpInterface<"TilingInfoOpInterface"> {
    let description = "Interface for operations to provide information about required/supported tiling configurations";

    let cppNamespace = "vpux::VPU";

    let methods = [
        InterfaceMethod<
            "Check, if the provided tiling configuration is supported by the operation implementation",
            "bool", "isSupportedTiling",
            (ins "const vpux::OutputTiling&":$tiles, "vpux::TilingMode":$tilingMode, "vpux::Logger":$log)
        >,
    ];

    let verify = [{
        return vpux::VPU::verifyTilingInfo($_op);
    }];
}

//
// EltwiseOp
//

// Single output element depends only on single input element.
// In other words - the iteration space is fully parallel across all indexes.
def VPU_EltwiseOp : NativeOpTrait<"EltwiseOp"> {
    let cppNamespace = "vpux::VPU";
}

//
// EMUUPAOpInterface
//

def VPU_EMUUPAOpInterface : OpInterface<"EMUUPAOpInterface"> {
    let description = "Interface for VPU ops that are serialized to UPA EMU Task";

    let cppNamespace = "vpux::VPU";

    let extraTraitClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::SHAVE_UPA;
        }
    }];
}

#endif
