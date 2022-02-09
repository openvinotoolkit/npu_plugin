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

#ifndef VPUX_COMPILER_DIALECT_IE_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_IE_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// LayerOpInterface
//

def IE_LayerOpInterface : OpInterface<"LayerOpInterface"> {
    let description = "Base interface for IE Layer Operation";

    let cppNamespace = "vpux::IE";

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

    let extraTraitClassDeclaration = [{
        static mlir::LogicalResult inferReturnTypes(mlir::MLIRContext* ctx, mlir::Optional<mlir::Location> loc,
                                                    mlir::ValueRange operands, mlir::DictionaryAttr attrs, mlir::RegionRange regions,
                                                    mlir::SmallVectorImpl<mlir::Type>& inferredTypes) {
            return vpux::IE::inferTensorTypes(ConcreteOp::inferReturnTypeComponents, ctx, loc, operands, attrs, regions, inferredTypes);
        }
    }];

    let verify = [{
        return vpux::IE::verifyLayer($_op);
    }];
}

//
// LayerWithPostOpInterface
//

def IE_LayerWithPostOpInterface : OpInterface<"LayerWithPostOpInterface"> {
    let description = "Interface for operations that support post-processing";

    let cppNamespace = "vpux::IE";

    let methods = [
        InterfaceMethod<
            "Get the post-processing operation",
            "llvm::Optional<mlir::OperationName>", "getPostOp", (ins),
            [{}],
            [{
                return vpux::IE::getLayerPostOp($_op);
            }]
        >,

        InterfaceMethod<
            "Get the post-processing operation attributes",
            "mlir::DictionaryAttr", "getPostOpAttrs", (ins),
            [{}],
            [{
                return vpux::IE::getLayerPostOpAttrs($_op);
            }]
        >,

        InterfaceMethod<
            "Set post-processing operation attribute",
            "void", "setPostOp", (ins "mlir::Operation*":$postOp),
            [{}],
            [{
                return vpux::IE::setLayerPostOp($_op, postOp);
            }]
        >,

        InterfaceMethod<
            "Set post-processing operation attribute",
            "void", "clearPostOp", (ins),
            [{}],
            [{
                return vpux::IE::clearLayerPostOp($_op);
            }]
        >,

        InterfaceMethod<
            "Set post-processing operation attribute",
            "bool", "isSupportedPostOp", (ins "mlir::Operation*":$postOp)
        >,
    ];
}

//
// AlignedChannelsOpInterface
//

def IE_AlignedChannelsOpInterface : OpInterface<"AlignedChannelsOpInterface"> {
    let description = "Interface for operations that require channel alignment";

    let cppNamespace = "vpux::IE";

    let methods = [
        InterfaceMethod<
            "Verify channel alignment",
            "mlir::LogicalResult", "verifyChannels", (ins)
        >,

        InterfaceMethod<
            "Get input channels alignment factor in elements",
            "int64_t", "getInputChannelAlignment", (ins)
        >,
        InterfaceMethod<
            "Get output channels alignment factor in elements",
            "int64_t", "getOutputChannelAlignment", (ins)
        >,

        InterfaceMethod<
            "Check HW channel number restrictions",
            "bool", "checkChannelRestrictions", (ins "int64_t":$channels)
        >
    ];
}

//
// LayoutInfoOpInterface
//

def IE_LayoutInfoOpInterface : OpInterface<"LayoutInfoOpInterface"> {
    let description = "Interface for operations to provide information about supported layout for inputs/outputs";

    let cppNamespace = "vpux::IE";

    let methods = [
        InterfaceMethod<
            "Infer supported Data Layouts from inputs to outputs or describe the supported combination",
            "void", "inferLayoutInfo",
            (ins "vpux::IE::LayerLayoutInfo&":$info)
        >,

        InterfaceMethod<
            "Get information about current layout for Layer inputs and outputs",
            "vpux::IE::LayerLayoutInfo", "getLayoutInfo", (ins),
            [{}],
            [{
                return vpux::IE::getLayoutInfo($_op);
            }]
        >
    ];
}

//
// TilingBuilderOpInterface
//

def IE_TilingBuilderOpInterface : OpInterface<"TilingBuilderOpInterface"> {
    let description = "Interface for operations to reify the provided tiling configuration";

    let cppNamespace = "vpux::IE";

    let methods = [
        InterfaceMethod<
            "Return the maximum number of tiles for each dimension",
            "SmallVector<int64_t>", "getMaxNumTiles", (ins),
            [{}],
            [{
                return vpux::IE::getMaxNumTiles($_op);
            }]
        >,

        InterfaceMethod<
            "Infer input tiles information from the output tile",
            "vpux::TilingInfo", "backInferTileInfo",
            (ins "const vpux::TileInfo&":$outputTile)
        >,

        InterfaceMethod<
            "Adjust the operation attributes to the current tile properties",
            "void", "adjustAttrs",
            (ins "const vpux::TilingInfo&":$inputTiling)
        >,
    ];
}

//
// TilingInfoOpInterface
//

def IE_TilingInfoOpInterface : OpInterface<"TilingInfoOpInterface"> {
    let description = "Interface for operations to provide information about required/supported tiling configurations";

    let cppNamespace = "vpux::IE";

    let methods = [
        InterfaceMethod<
            "Check, if the provided tiling configuration is supported by the operation implementation",
            "bool", "isSupportedTiling",
            (ins "const vpux::OutputTiling&":$tiles, "vpux::Logger":$log)
        >,

        InterfaceMethod<
            "Check, if the operation support prefetch tiling with cmx memory checking only",
            "bool", "isSupportedPrefetchTiling",
            (ins "ShapeRef":$tileAxis, "vpux::Logger":$log)
        >,

        InterfaceMethod<
            "Check, if the pattern (current op and parent op, with specific tiling) supports prefetch with cmx memory checking only",
            "bool", "isSupportedPrefetchPattern",
            (ins "ShapeRef":$tileAxis, "mlir::Operation*":$parentOp, "ShapeRef":$parentTileAxis, "vpux::Logger":$log)
        >,
    ];

    let verify = [{
        return vpux::IE::verifyTilingInfo($_op);
    }];
}

//
// ResourceOpBase
//

def IERT_ResourceOpInterface : OpInterface<"ResourceOpInterface"> {
    let description = "Base interface for for a meta-like operation containing information about a runtime resource";

    let cppNamespace = "vpux::IERT";

    let methods = [
        InterfaceMethod<
            "Get all layer input memory buffers",
            "mlir::StringAttr", "getKind", (ins),
            [{}],
            [{
                return $_op.sym_nameAttr();
            }]
        >,
    ];

    let extraTraitClassDeclaration = [{
         template<class ConcreteAttr>
         ConcreteAttr getKindAs() {
             VPUX_THROW_UNLESS(getKind().template isa<ConcreteAttr>(), "Resource type is not {0}", typeid(ConcreteAttr).name());
             return getKind().template cast<ConcreteAttr>();
         }

         template<class ConcreteAttr>
         typename ConcreteAttr::ValueType getKindValue() {
             VPUX_THROW_UNLESS(getKind().template isa<ConcreteAttr>(), "Resource type is not {0}", typeid(ConcreteAttr).name());
             return getKind().template cast<ConcreteAttr>().getValue();
         }
    }];
}

//
// EltwiseOp
//

// Single output element depends only on single input element.
// In other words - the iteration space is fully parallel across all indexes.
def IE_EltwiseOp : NativeOpTrait<"EltwiseOp"> {
    let cppNamespace = "vpux::IE";
}

//
// ElemTypeInfoOpInterface
//

// using ElemTypeInfo = LayerDataInfo<mlir::Type>;

def IE_ElemTypeInfoOpInterface : OpInterface<"ElemTypeInfoOpInterface"> {

    let cppNamespace = "vpux::IE";
    let methods = [
        InterfaceMethod<"Infer element type down",
            "void", "inferElemTypeInfo",
            (ins "vpux::IE::LayerDataInfo<mlir::Type>&":$info)
        >,
        InterfaceMethod<"Infer element type up",
            "void", "inferElemTypeInfoUp",
            (ins "vpux::IE::LayerDataInfo<mlir::Type>&":$info)
        >,
        InterfaceMethod<"Get element types for inputs and outpus",
            "vpux::IE::LayerDataInfo<mlir::Type>", "getElemTypeInfo", (ins),
            [{}],
            [{
                return vpux::IE::getElemTypeInfo($_op);
            }]
        >
    ];
}

#endif
