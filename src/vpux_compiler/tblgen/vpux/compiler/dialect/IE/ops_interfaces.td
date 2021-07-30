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
        >,

        InterfaceMethod<
            "Get information about current layout for Layer inputs and outputs",
            "vpux::IE::DataOrderInfo", "getDataOrderInfo", (ins),
            [{}],
            [{
                return vpux::IE::getLayerDataOrderInfo($_op);
            }]
        >,
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
            "vpux::Optional<vpux::IE::PostOp>", "post_op", (ins)
        >,

        InterfaceMethod<
            "Set post-processing operation attribute",
            "void", "post_opAttr", (ins "vpux::IE::PostOp":$postOp)
        >,
    ];
}

#endif
