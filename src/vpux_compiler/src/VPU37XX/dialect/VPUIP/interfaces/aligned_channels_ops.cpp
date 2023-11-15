//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU37XX/dialect/VPUIP/ops_interfaces.hpp"

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"

using namespace vpux;

namespace {

template <class MainOpType>
class AlignedChannelsOpModel37XX final :
        public IE::AlignedChannelsOpInterface::ExternalModel<AlignedChannelsOpModel37XX<MainOpType>, MainOpType> {
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
            const auto inOrder = inputType.getDimsOrder();
            if (inOrder == DimsOrder::NCHW) {
                // C-major convolution has no specific requirements
                return 1;
            }

            const auto inputC = inputType.getShape()[Dims4D::Act::C];
            // For MatMul converted operation to IE.Convolution, weights const operation is moved to
            // first input of convolution and we don't support them as compress conv.
            bool filterIsConstOp = op->getOperand(1).getDefiningOp<Const::DeclareOp>() != nullptr;
            if (inputC == VPU::NCEInvariant::VPU_COMPRESSED_INPUT_CHANNEL_NUM && filterIsConstOp) {
                return VPU::NCEInvariant::VPU_COMPRESSED_INPUT_CHANNEL_NUM;
            }

            const auto weightsType = op->getOperand(1).getType().cast<vpux::NDTypeInterface>();
            const bool isFP16 = inputType.getElementType().isF16() || weightsType.getElementType().isF16();
            if (!isFP16 && inputC < VPU::NCEInvariant::VPU_COMPRESSED_INPUT_CHANNEL_NUM && filterIsConstOp) {
                return VPU::NCEInvariant::VPU_COMPRESSED_INPUT_CHANNEL_NUM;
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

void vpux::VPUIP::arch37xx::registerAlignedChannelsOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*) {
        IE::ConvolutionOp::attachInterface<AlignedChannelsOpModel37XX<IE::ConvolutionOp>>(*ctx);
        IE::GroupConvolutionOp::attachInterface<AlignedChannelsOpModel37XX<IE::GroupConvolutionOp>>(*ctx);
        IE::MaxPoolOp::attachInterface<AlignedChannelsOpModel37XX<IE::MaxPoolOp>>(*ctx);
        IE::AvgPoolOp::attachInterface<AlignedChannelsOpModel37XX<IE::AvgPoolOp>>(*ctx);
        IE::AddOp::attachInterface<AlignedChannelsOpModel37XX<IE::AddOp>>(*ctx);
        IE::MultiplyOp::attachInterface<AlignedChannelsOpModel37XX<IE::MultiplyOp>>(*ctx);
        IE::SubtractOp::attachInterface<AlignedChannelsOpModel37XX<IE::SubtractOp>>(*ctx);
        IE::AndOp::attachInterface<AlignedChannelsOpModel37XX<IE::AndOp>>(*ctx);
        IE::InterpolateOp::attachInterface<AlignedChannelsOpModel37XX<IE::InterpolateOp>>(*ctx);
    });
}
