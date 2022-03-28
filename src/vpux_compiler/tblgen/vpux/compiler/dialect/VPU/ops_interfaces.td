//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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
        if (vpux::VPU::details::validatePrecisionForNCE($_op).failed()) {
            return mlir::failure();
        }

        if (vpux::VPU::details::validateWorkloadsRegion($_op->getLoc(), mlir::cast<ConcreteOp>($_op).workloads()).failed()) {
            return mlir::failure();
        }

        return mlir::cast<ConcreteOp>($_op).verifyChannels();
    }];
}

#endif
