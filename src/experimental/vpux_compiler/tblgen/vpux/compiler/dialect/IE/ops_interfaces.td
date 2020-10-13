//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#ifndef VPUX_COMPILER_DIALECT_IE_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_IE_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// NetworkInformation
//

def IE_NetworkInformation : NativeOpTrait<"NetworkInformation"> {
    let cppNamespace = "vpux::IE";
}

//
// SoftMaxLayerInterface
//

def IE_SoftMaxLayerInterface : OpInterface<"SoftMaxLayerInterface"> {
    let description = "Interface for SoftMax Layer";

    let cppNamespace = "vpux::IE";

    let methods = [
        InterfaceMethod<
            "Get the input tensor logical shape",
            "vpux::ShapeRef", "getInputShape", (ins)
        >,

        InterfaceMethod<
            "Get the axis parameter in logical dimensions",
            "vpux::Dim", "getAxisDim", (ins)
        >,
    ];

    let verify = [{
        return vpux::IE::verifySoftMaxLayer($_op);
    }];
}

#endif
