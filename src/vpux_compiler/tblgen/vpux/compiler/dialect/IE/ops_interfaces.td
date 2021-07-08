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
// IELayer
//

def IE_Layer : NativeOpTrait<"IELayer"> {
    let cppNamespace = "vpux::IE";
}

//
// MultiLayerInterface
//

def MultiLayerInterface : OpInterface<"MultiLayerInterface"> {
    let description = "Interface for operations that support post-processing";

    let cppNamespace = "vpux::IE";

    let methods = [
        InterfaceMethod<
            "Get the post-processing operation",
            "vpux::IE::PostOp", "getPostOp", (ins),
            [{
                return $_op.post_opAttr();
            }]
        >,

        InterfaceMethod<
            "Set post-processing operation attribute",
            "void", "setPostOp", (ins "vpux::IE::PostOp":$postOp),
            [{
                $_op.post_opAttr(postOp);
            }]
        >,
    ];
}

#endif
