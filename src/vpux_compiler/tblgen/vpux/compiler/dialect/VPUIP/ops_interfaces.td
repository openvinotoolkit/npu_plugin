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

#ifndef VPUX_COMPILER_DIALECT_VPUIP_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_VPUIP_OPS_INTERFACES

include "mlir/IR/OpBase.td"

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
        vpux::IndexedSymbolAttr getExecutor(uint32_t& numUnits) {
            return vpux::VPUIP::getTaskOpExecutor($_op, numUnits);
        }
    }];
}

//
// UPATaskOpInterface
//

def VPUIP_UPATaskOpInterface : OpInterface<"UPATaskOpInterface"> {
    let description = "Interface for VPUIP UPA Task";

    let cppNamespace = "vpux::VPUIP";

    let methods = [
        InterfaceMethod<
            "Get maximal number of UPA SHAVEs to use",
            "vpux::Optional<int64_t>", "maxShaves", (ins),
            [{
                return $_op.maxShaves();
            }]
        >,
        InterfaceMethod<
            "Update maximal number of UPA SHAVEs to use",
            "void", "setMaxShaves", (ins "int64_t":$maxShaves),
            [{}],
            [{
                $_op.maxShavesAttr(vpux::getIntAttr($_op.getContext(), maxShaves));
            }]
        >,
    ];

    let verify = [{
        return vpux::VPUIP::verifyUPATask($_op);
    }];

    let extraTraitClassDeclaration = [{
        static vpux::VPU::ExecutorKind getExecutorKind() {
            return vpux::VPU::ExecutorKind::SHAVE_UPA;
        }
    }];
}

//
// Legacy4D
//

def VPUIP_Legacy4D : NativeOpTrait<"Legacy4D"> {
    let cppNamespace = "vpux::VPUIP";
}

#endif
