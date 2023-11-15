//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU30XX/dialect/VPUIP/ops_interfaces.hpp"

#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"

using namespace vpux;

namespace {

template <class MainOpType>
class AlignedChannelsOpModel30XX final :
        public IE::AlignedChannelsOpInterface::ExternalModel<AlignedChannelsOpModel30XX<MainOpType>, MainOpType> {
public:
    mlir::LogicalResult verifyChannels(mlir::Operation* op) const {
        if (!canBeExecutedOnNCE(op)) {
            // SW version of the operation has no specific requirements
            return mlir::success();
        }

        return VPUIP::NCEInvariant::verifyChannels(mlir::cast<MainOpType>(op));
    }

    int64_t getInputChannelAlignment(mlir::Operation* op) const {
        if (!canBeExecutedOnNCE(op)) {
            // SW version of the operation has no specific requirements
            return 1;
        }

        const auto inputType = op->getOperand(0).getType().cast<vpux::NDTypeInterface>();
        if (mlir::isa<IE::ConvolutionOp>(op)) {
            if (DimsOrder::NCHW == inputType.getDimsOrder()) {
                // C-major convolution has no specific requirements
                return 1;
            }
        }

        return VPU::NCEInvariant::getAlignment(inputType.getElementType());
    }

    int64_t getOutputChannelAlignment(mlir::Operation* op) const {
        if (!canBeExecutedOnNCE(op)) {
            // SW version of the operation has no specific requirements
            return 1;
        }
        const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
        return VPU::NCEInvariant::getAlignment(outputType.getElementType());
    }

private:
    static bool canBeExecutedOnNCE(mlir::Operation* op) {
        if (VPU::getCompilationMode(op) == VPU::CompilationMode::ReferenceSW) {
            // We are in reference SW compilation mode
            return false;
        }

        if (VPU::NCEInvariant::isSupported(mlir::cast<MainOpType>(op)).failed()) {
            // Basic NCE invariants check failed, the operation will fallback to SW mode
            return false;
        }
        return true;
    }
};

}  // namespace

void vpux::VPUIP::arch30xx::registerAlignedChannelsOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*) {
        IE::ConvolutionOp::attachInterface<AlignedChannelsOpModel30XX<IE::ConvolutionOp>>(*ctx);
        IE::GroupConvolutionOp::attachInterface<AlignedChannelsOpModel30XX<IE::GroupConvolutionOp>>(*ctx);
        IE::MaxPoolOp::attachInterface<AlignedChannelsOpModel30XX<IE::MaxPoolOp>>(*ctx);
        IE::AvgPoolOp::attachInterface<AlignedChannelsOpModel30XX<IE::AvgPoolOp>>(*ctx);
        IE::AddOp::attachInterface<AlignedChannelsOpModel30XX<IE::AddOp>>(*ctx);
        IE::MultiplyOp::attachInterface<AlignedChannelsOpModel30XX<IE::MultiplyOp>>(*ctx);
        IE::SubtractOp::attachInterface<AlignedChannelsOpModel30XX<IE::SubtractOp>>(*ctx);
        IE::AndOp::attachInterface<AlignedChannelsOpModel30XX<IE::AndOp>>(*ctx);
        IE::InterpolateOp::attachInterface<AlignedChannelsOpModel30XX<IE::InterpolateOp>>(*ctx);
    });
}
