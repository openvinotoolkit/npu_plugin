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

#ifndef VPUX_COMPILER_DIALECT_VPU_OPS_INTERFACES
#define VPUX_COMPILER_DIALECT_VPU_OPS_INTERFACES

include "mlir/IR/OpBase.td"

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

        static int64_t getInputChannelAlignment(vpux::NDTypeInterface inputType) {
            return vpux::VPU::NCEInvariant::getAlignment(inputType.getElementType());
        }
        static int64_t getOutputChannelAlignment(vpux::NDTypeInterface outputType) {
            return vpux::VPU::NCEInvariant::getAlignment(outputType.getElementType());
        }

        int64_t getInputChannelAlignment() {
            const auto inputType = $_op->getOperand(0).getType().template cast<vpux::NDTypeInterface>();
            return ConcreteOp::getInputChannelAlignment(inputType);
        }
        int64_t getOutputChannelAlignment() {
            const auto outputType = $_op->getResult(0).getType().template cast<vpux::NDTypeInterface>();
            return ConcreteOp::getOutputChannelAlignment(outputType);
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

        return mlir::success();
    }];
}

#endif
