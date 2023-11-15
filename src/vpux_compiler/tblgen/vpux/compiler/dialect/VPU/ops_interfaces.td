//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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
// ClusteredOpInterface
//

def VPU_ClusteredOpInterface : OpInterface<"ClusteredOpInterface"> {
    let description = "Interface for clustered operations";
    let cppNamespace = "vpux::VPU";
    let methods = [
        InterfaceMethod<
                "Check layer strategy compatibility",
                "bool", "checkStrategyCompatibility",
                (ins "VPU::MultiClusterStrategy":$strategy)
                >,
        InterfaceMethod<
                "Get multi-cluster strategy",
                "llvm::Optional<vpux::VPU::MultiClusterStrategy>", "getMultiClusterStrategy", (ins),
                [{
                    return $_op.getMultiClusterStrategy();
                }]
                >,

        InterfaceMethod<
                "Set multi-cluster strategy attribute",
                "void", "setMultiClusterStrategy", (ins "vpux::VPU::MultiClusterStrategy":$strategy),
                [{}],
                [{
                    return vpux::VPU::setLayerMultiClusterStrategy($_op, strategy);
                }]
                >,

        InterfaceMethod<
                "Get DistributedTensorAttr with per cluster shapes/offsets defined explicitly, rather than through params such as kernel/pad/stride etc.",
                "vpux::VPU::DistributedTensorAttr", "getExplicitDistributedTensorAttr",
                (ins "vpux::ShapeRef":$shape, "vpux::VPU::DistributionMode":$distributionMode,
                "mlir::ArrayAttr":$numTiles, "mlir::IntegerAttr":$numClusters, "mlir::ArrayAttr":$alignment,
                "mlir::ArrayAttr":$kernel, "vpux::VPU::PaddingAttr":$pad, "mlir::ArrayAttr":$stride,
                "mlir::UnitAttr":$uniformDistributedSegments)
                >
    ];
}

//
// SWOpInterface
//

def VPU_SWOpInterface : OpInterface<"SWOpInterface"> {
    let description = "Interface for SW operations";

    let cppNamespace = "vpux::VPU";

    let methods = [
        InterfaceMethod<
            "Check if layer fits in NNCMX excluding reserved memory",
            "bool", "fitIntoCMX", (ins "::llvm::ArrayRef<vpux::NDTypeInterface>":$buffers, "Byte":$reservedMem),
            [{
                return $_op.fitIntoCMX(buffers, reservedMem);
            }]
            >,
        InterfaceMethod<
            "Check if layer is supported by cost model for cycle cost calculation",
            "bool", "supportCycleCostCalculation", (ins)
            >
    ];
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
        >
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
            "Get NCE operation kernel size",
            "SmallVector<int64_t>", "getKernelSizeVal", (ins)
        >,

        InterfaceMethod<
            "Get NCE operation kernel strides",
            "SmallVector<int64_t>", "getStridesVal", (ins)
        >,

        InterfaceMethod<
            "Get NCE operation padding",
            "vpux::VPU::PaddingAttr", "getPad", (ins)
        >,

        InterfaceMethod<
            "Get NCE operation PPE",
            "llvm::Optional<vpux::VPU::PPETaskAttr>", "getPPE", (ins),
            [{
                return $_op.getPpe();
            }]
        >,

        InterfaceMethod<
            "Set NCE operation PPE",
            "void", "setPPE", (ins "vpux::VPU::PPETaskAttr":$ppe_task_attr),
            [{
                $_op->setAttr($_op.getPpeAttrName(), ppe_task_attr);
            }]
        >,

        InterfaceMethod<
            "Get the weights operand of the operation",
            "mlir::Value", "getWeightsOperand", (ins),
            [{}],
            [{
                return nullptr;
            }]
        >,

        InterfaceMethod<
            "Get the weights table operand of the operation",
            "mlir::Value", "getWeightsTableOperand", (ins),
            [{}],
            [{
                return nullptr;
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
        >,


        InterfaceMethod<
            "verify input type",
            "mlir::LogicalResult", "verifyInputType",
            (ins "vpux::NDTypeInterface":$inputType)
        >

    ];

    let extraTraitClassDeclaration = [{
        mlir::LogicalResult verifyChannels() {
            auto arch = getArch($_op);
            return mlir::success(
                vpux::VPU::NCEInvariant::isInputActTypeSupported(arch,
                    $_op->getOperand(0).getType().template cast<vpux::NDTypeInterface>(),
                    mlir::cast<ConcreteOp>($_op).getInputChannelAlignment(), false) &&
                vpux::VPU::NCEInvariant::isOutputActTypeSupported(
                    $_op->getResult(0).getType().template cast<vpux::NDTypeInterface>(),
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
        return vpux::VPU::verifyNCEOp($_op);
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
            "Infer output tiles information for all outputs based on the first output tile",
            "vpux::OutputTiling", "getOutputTiling",
            (ins "const vpux::TileInfo&":$outputTile, "vpux::Logger":$log),
            [{}],
            [{
                return vpux::OutputTiling({outputTile});
            }]
        >,

        InterfaceMethod<
            "Adjust the operation attributes to the current tile properties",
            "void", "adjustAttrs",
            (ins "const vpux::TilingInfo&":$inputTiling, "const TileInfo&":$outputTile)
        >,

        InterfaceMethod<
            "Get optimal output tiling scheme",
            "mlir::FailureOr<vpux::OutputTiling>", "getTilingStrategy",
            (ins "vpux::TilingMode":$tilingMode, "vpux::Logger":$log)
        >
    ];
}

//
// VerticalFusionOpInterface
//

def VPU_VerticalFusionOpInterface : OpInterface<"VerticalFusionOpInterface"> {
    let description = "Interface for operations which might be vertically fused";

    let cppNamespace = "vpux::VPU";

    let methods = [
        InterfaceMethod<
            "Check if VF supported for operation",
            "bool", "isVFSupported", (ins),
            [{}],
            [{
                return $_op->hasAttr("tilingStrategy");
            }]
        >,

        InterfaceMethod<
            "Check if operation can be fused as single operation in VF block with next block",
            "bool", "availableSingleMerge", (ins),
            [{}],
            [{
                return true;
            }]
        >
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

//
// ViewLikeOpInterface
//

def VPU_ViewLikeOpInterface: OpInterface<"ViewLikeOpInterface"> {
    let description = "Interface for VPU view-like ops";

    let cppNamespace = "vpux::VPU";
}

//
// GroupedViewLikeOpInterface
//

def VPU_GroupedViewLikeOpInterface: OpInterface<"GroupedViewLikeOpInterface"> {
    let description = "Interface for grouped VPU view-like ops";

    let cppNamespace = "vpux::VPU";
}

//
// LimitedToArch op trait
//

class LimitedToArch<list<string> archs>
    : ParamNativeOpTrait<"LimitedToArch", !interleave(archs, ", ")>  {
    let cppNamespace = "vpux::VPU";
}

#endif
