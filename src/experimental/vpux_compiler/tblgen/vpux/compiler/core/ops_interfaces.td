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

#ifndef VPUX_COMPILER_CORE_OPS_INTERFACES
#define VPUX_COMPILER_CORE_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// DataInfoInterface
//

def DataInfoInterface : OpInterface<"DataInfoInterface"> {
    let description = "Interface for Operation that represents Network Data information";

    let cppNamespace = "vpux";

    let methods = [
        InterfaceMethod<
            "Get the name of the Data",
            "vpux::StringRef", "name", (ins),
            [{
                return $_op.name();
            }]
        >,

        InterfaceMethod<
            "Get the precision (element type) of the Data",
            "mlir::Type", "precision", (ins),
            [{
                return $_op.precision();
            }]
        >,

        InterfaceMethod<
            "Get the layout (DimsOrder) of the Data",
            "vpux::DimsOrder", "getDimsOrder", (ins)
        >,
    ];

    let verify = [{
        return vpux::details::verifyDataInfo($_op);
    }];
}

//
// NetInfoInterface
//

def NetInfoInterface : OpInterface<"NetInfoInterface"> {
    let description = "Interface for Operation that represents Network information";

    let cppNamespace = "vpux";

    let methods = [
         InterfaceMethod<
             "Get the entry point for the network",
             "mlir::FlatSymbolRefAttr", "entryPointAttr", (ins),
             [{
                 return $_op.entryPointAttr();
             }]
         >,
        InterfaceMethod<
            "Get the entry point for the network",
            "vpux::StringRef", "entryPoint", (ins),
            [{
                return $_op.entryPoint();
            }]
        >,

        InterfaceMethod<
            "Get the Region with information about network inputs",
            "mlir::Region&", "inputsInfo", (ins),
            [{
                return $_op.inputsInfo();
            }]
        >,
        InterfaceMethod<
            "Get the information about network inputs",
            "vpux::SmallVector<vpux::DataInfoInterface, 1>",
            "getInputsInfo", (ins),
            [{}],
            [{
                return vpux::details::getDataInfoVec($_op.inputsInfo());
            }]
        >,

        InterfaceMethod<
            "Get the Region with information about network outputs",
            "mlir::Region&", "outputsInfo", (ins),
            [{
                return $_op.outputsInfo();
            }]
        >,
        InterfaceMethod<
            "Get the information about network outputs",
            "vpux::SmallVector<vpux::DataInfoInterface, 1>",
            "getOutputsInfo", (ins),
            [{}],
            [{
                return vpux::details::getDataInfoVec($_op.outputsInfo());
            }]
        >,
    ];

    let verify = [{
        return vpux::details::verifyNetInfo($_op);
    }];

    let extraTraitClassDeclaration = [{
        static mlir::LogicalResult getFromModule(
                mlir::ModuleOp module,
                ConcreteOp& netInfo,
                mlir::FuncOp& netFunc) {
            return vpux::details::getNetInfo(module, netInfo, netFunc);
        }
    }];
}

#endif
