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

#ifndef VPUX_COMPILER_CORE_OPS_INTERFACES
#define VPUX_COMPILER_CORE_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// MultiViewOpInterface
//

def MultiViewOpInterface : OpInterface<"MultiViewOpInterface"> {
    let description = [{
        An extended version of view-like operation.
        It allows to define several resulting views in single operation.
    }];

    let cppNamespace = "vpux";

    let methods = [
        InterfaceMethod<
            "The source buffer from which the corresponding view is created, NULL if the result is not a view",
            "mlir::Value", "getViewSource", (ins "ptrdiff_t":$resultInd)
        >
    ];
}

def DotInterface : OpInterface<"DotInterface"> {
    let description = "Base interface for Dot graph generation";

    let cppNamespace = "vpux";

    let methods = [
        InterfaceMethod<
            "Get node color",
            "vpux::DotNodeColor", "getNodeColor", (ins),
            [{}],
            [{return vpux::DotNodeColor::NONE;}]
        >,

        InterfaceMethod<
            "Node type is a declaration",
            "bool", "isDeclaration", (ins),
            [{}],
            [{return false;}]
        >,

        InterfaceMethod<
            "If non empty overrides the default attributes in the dot graph",
            "std::string", "printAttributes", (ins),
            [{}],
            [{return "";}]
        >
    ];
}

#endif
