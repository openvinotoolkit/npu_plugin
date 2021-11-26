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

#ifndef VPUX_COMPILER_DIALECT_EMU_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_EMU_OPS_INTERFACES

include "mlir/IR/OpBase.td"

//
// TaskOpInterface
//

def EMU_TaskOpInterface : OpInterface<"TaskOpInterface"> {
    let description = "Interface for EMU Task";

    let cppNamespace = "vpux::EMU";

    let methods = [

        InterfaceMethod<
            "Serialize the Task to BLOB format",
            "vpux::EMU::BlobWriter::SpecificTask",
            "serialize", (ins "vpux::EMU::BlobWriter&":$writer)
        >,

        StaticInterfaceMethod<
            "Get the EMU TaskType for the Operation",
            "vpux::EMU::TaskType", "getTaskType", (ins),
            [{
                return ConcreteOp::getTaskType();
            }]
        >,
    ];

}

//
// UPATaskOpInterface
//

def EMU_UPATaskOpInterface : OpInterface<"UPATaskOpInterface"> {
    let description = "Interface for EMU UPA Task";

    let cppNamespace = "vpux::EMU";

    let verify = [{
        return vpux::EMU::verifyUPATask($_op);
    }];

    let extraTraitClassDeclaration = [{
        static vpux::EMU::TaskType getTaskType() {
            return vpux::EMU::TaskType::UPA;
        }
    }];
}

#endif
