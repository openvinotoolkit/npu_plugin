//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/utils/asm.hpp"

using namespace vpux;

mlir::LogicalResult VPU::sameLayout(VPU::DistributedTensorType inDistributedType,
                                    VPU::DistributedTensorType outDistributedType, LogCb logCb) {
    if (inDistributedType.getOrder() != outDistributedType.getOrder()) {
        logCb(formatv("Mismatch between order for input ({0}) and output ({1}).", inDistributedType.getOrder(),
                      outDistributedType.getOrder()));
        return mlir::failure();
    }
    return mlir::success();
}

mlir::LogicalResult VPU::sameLayout(VPUIP::DistributedBufferType inDistributedType,
                                    VPUIP::DistributedBufferType outDistributedType, LogCb logCb) {
    auto isContinuousWithSameOrder = [&]() {
        const auto inStrideReqs = StrideReqs::compact(inDistributedType.getShape().size());
        const auto outStrideReqs = StrideReqs::compact(outDistributedType.getShape().size());
        auto inRes = inStrideReqs.checkStrides(inDistributedType);
        auto outRes = outStrideReqs.checkStrides(outDistributedType);
        return inRes && outRes && inDistributedType.getDimsOrder() == outDistributedType.getDimsOrder();
    };

    // The strides will be checked when comparing the layouts. So the function will return true if the layouts are equal
    // or the buffers are compact with same dim order
    if (inDistributedType.getLayout() != outDistributedType.getLayout() && !isContinuousWithSameOrder()) {
        logCb(formatv("Mismatch between order for input ({0}) and output ({1}).", inDistributedType.getLayout(),
                      outDistributedType.getLayout()));
        return mlir::failure();
    }
    return mlir::success();
}

bool VPU::isVFNCESupported(VPU::NCEOpInterface op) {
    auto isOne = [](auto val) {
        return val == 1;
    };

    if (llvm::all_of(op.getStrides(), isOne)) {
        return true;
    }

    return false;
}

//
// materializeConstant
//

mlir::Operation* vpux::VPU::VPUDialect::materializeConstant(mlir::OpBuilder& builder, mlir::Attribute value,
                                                            mlir::Type type, mlir::Location loc) {
    if (!value.isa<Const::ContentAttr>()) {
        (void)errorAt(loc, "Can't materialize VPU Constant from Attribute '{0}'", value);
        return nullptr;
    }

    if (!type.isa<mlir::RankedTensorType>()) {
        (void)errorAt(loc, "Can't materialize VPU Constant for Type '{0}'", type);
        return nullptr;
    }

    return builder.create<Const::DeclareOp>(loc, type, value.cast<Const::ContentAttr>());
}

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/generated/ops.cpp.inc>

namespace {

//
// LayoutInfoOpModel
//

template <class ImplOpType>
class LayoutInfoOpModelForSW final :
        public IE::LayoutInfoOpInterface::FallbackModel<LayoutInfoOpModelForSW<ImplOpType>> {
public:
    void inferLayoutInfo(mlir::Operation* origOp, IE::LayerLayoutInfo& info) const {
        ImplOpType::inferLayoutInfo(origOp, info);
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

//
// redirectOpInterfacesForIE
//

template <template <class> class OpModelForSW>
void redirectOpInterfacesForIE(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, IE::IEDialect*) {
        // VPU_SameInOutDimsOrder
        IE::ConvertOp::attachInterface<OpModelForSW<VPU::ConvertOp>>(*ctx);
        IE::SoftMaxOp::attachInterface<OpModelForSW<VPU::SoftMaxOp>>(*ctx);
        IE::AvgPoolOp::attachInterface<OpModelForSW<VPU::AvgPoolOp>>(*ctx);
        IE::AdaptiveAvgPoolOp::attachInterface<OpModelForSW<VPU::AdaptiveAvgPoolOp>>(*ctx);
        IE::AdaptiveMaxPoolOp::attachInterface<OpModelForSW<VPU::AdaptiveMaxPoolOp>>(*ctx);
        IE::RollOp::attachInterface<OpModelForSW<VPU::RollOp>>(*ctx);
        IE::TanOp::attachInterface<OpModelForSW<VPU::TanOp>>(*ctx);
        IE::SinOp::attachInterface<OpModelForSW<VPU::SinOp>>(*ctx);
        IE::CosOp::attachInterface<OpModelForSW<VPU::CosOp>>(*ctx);
        IE::LogOp::attachInterface<OpModelForSW<VPU::LogOp>>(*ctx);
        IE::SeluOp::attachInterface<OpModelForSW<VPU::SeluOp>>(*ctx);
        IE::PReluOp::attachInterface<OpModelForSW<VPU::PReluOp>>(*ctx);
        IE::SelectOp::attachInterface<OpModelForSW<VPU::SelectOp>>(*ctx);
        IE::SwishOp::attachInterface<OpModelForSW<VPU::SwishOp>>(*ctx);
        IE::LeakyReluOp::attachInterface<OpModelForSW<VPU::LeakyReluOp>>(*ctx);
        IE::HardSigmoidOp::attachInterface<OpModelForSW<VPU::HardSigmoidOp>>(*ctx);
        IE::GridSampleOp::attachInterface<OpModelForSW<VPU::GridSampleOp>>(*ctx);
        IE::BucketizeOp::attachInterface<OpModelForSW<VPU::BucketizeOp>>(*ctx);
        IE::TileOp::attachInterface<OpModelForSW<VPU::TileOp>>(*ctx);
        IE::PerAxisTileOp::attachInterface<OpModelForSW<VPU::PerAxisTileOp>>(*ctx);
        IE::NegativeOp::attachInterface<OpModelForSW<VPU::NegativeOp>>(*ctx);
        IE::PadOp::attachInterface<OpModelForSW<VPU::PadOp>>(*ctx);
        IE::LSTMCellOp::attachInterface<OpModelForSW<VPU::LSTMCellOp>>(*ctx);
        IE::SpaceToDepthOp::attachInterface<OpModelForSW<VPU::SpaceToDepthOp>>(*ctx);
        IE::DepthToSpaceOp::attachInterface<OpModelForSW<VPU::DepthToSpaceOp>>(*ctx);
        IE::NormalizeL2Op::attachInterface<OpModelForSW<VPU::NormalizeL2Op>>(*ctx);
        IE::NormalizeIEOp::attachInterface<OpModelForSW<VPU::NormalizeIEOp>>(*ctx);
        IE::CumSumOp::attachInterface<OpModelForSW<VPU::CumSumOp>>(*ctx);
        IE::ReverseSequenceOp::attachInterface<OpModelForSW<VPU::ReverseSequenceOp>>(*ctx);
        IE::SoftPlusOp::attachInterface<OpModelForSW<VPU::SoftPlusOp>>(*ctx);

        // VPU_SameInOutDimsOrder
        IE::ConvertOp::attachInterface<OpModelForSW<VPU::ConvertOp>>(*ctx);
        IE::SoftMaxOp::attachInterface<OpModelForSW<VPU::SoftMaxOp>>(*ctx);
        IE::AvgPoolOp::attachInterface<OpModelForSW<VPU::AvgPoolOp>>(*ctx);
        IE::AdaptiveAvgPoolOp::attachInterface<OpModelForSW<VPU::AdaptiveAvgPoolOp>>(*ctx);
        IE::AdaptiveMaxPoolOp::attachInterface<OpModelForSW<VPU::AdaptiveMaxPoolOp>>(*ctx);
        IE::RollOp::attachInterface<OpModelForSW<VPU::RollOp>>(*ctx);
        IE::TanOp::attachInterface<OpModelForSW<VPU::TanOp>>(*ctx);
        IE::SinOp::attachInterface<OpModelForSW<VPU::SinOp>>(*ctx);
        IE::CosOp::attachInterface<OpModelForSW<VPU::CosOp>>(*ctx);
        IE::LogOp::attachInterface<OpModelForSW<VPU::LogOp>>(*ctx);
        IE::SeluOp::attachInterface<OpModelForSW<VPU::SeluOp>>(*ctx);
        IE::PReluOp::attachInterface<OpModelForSW<VPU::PReluOp>>(*ctx);
        IE::SelectOp::attachInterface<OpModelForSW<VPU::SelectOp>>(*ctx);
        IE::SwishOp::attachInterface<OpModelForSW<VPU::SwishOp>>(*ctx);
        IE::LeakyReluOp::attachInterface<OpModelForSW<VPU::LeakyReluOp>>(*ctx);
        IE::HardSigmoidOp::attachInterface<OpModelForSW<VPU::HardSigmoidOp>>(*ctx);
        IE::GridSampleOp::attachInterface<OpModelForSW<VPU::GridSampleOp>>(*ctx);
        IE::BucketizeOp::attachInterface<OpModelForSW<VPU::BucketizeOp>>(*ctx);
        IE::TileOp::attachInterface<OpModelForSW<VPU::TileOp>>(*ctx);
        IE::PerAxisTileOp::attachInterface<OpModelForSW<VPU::PerAxisTileOp>>(*ctx);
        IE::NegativeOp::attachInterface<OpModelForSW<VPU::NegativeOp>>(*ctx);
        IE::PadOp::attachInterface<OpModelForSW<VPU::PadOp>>(*ctx);
        IE::LSTMCellOp::attachInterface<OpModelForSW<VPU::LSTMCellOp>>(*ctx);
        IE::SpaceToDepthOp::attachInterface<OpModelForSW<VPU::SpaceToDepthOp>>(*ctx);
        IE::DepthToSpaceOp::attachInterface<OpModelForSW<VPU::DepthToSpaceOp>>(*ctx);
        IE::NormalizeL2Op::attachInterface<OpModelForSW<VPU::NormalizeL2Op>>(*ctx);
        IE::NormalizeIEOp::attachInterface<OpModelForSW<VPU::NormalizeIEOp>>(*ctx);
        IE::CumSumOp::attachInterface<OpModelForSW<VPU::CumSumOp>>(*ctx);
        IE::ReverseSequenceOp::attachInterface<OpModelForSW<VPU::ReverseSequenceOp>>(*ctx);
        IE::SoftPlusOp::attachInterface<OpModelForSW<VPU::SoftPlusOp>>(*ctx);
        IE::DFTOp::attachInterface<OpModelForSW<VPU::DFTOp>>(*ctx);
        IE::RDFTOp::attachInterface<OpModelForSW<VPU::RDFTOp>>(*ctx);
        IE::IDFTOp::attachInterface<OpModelForSW<VPU::IDFTOp>>(*ctx);
        IE::IRDFTOp::attachInterface<OpModelForSW<VPU::IRDFTOp>>(*ctx);

        // VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
        IE::SigmoidOp::attachInterface<OpModelForSW<VPU::SigmoidOp>>(*ctx);
        IE::ClampOp::attachInterface<OpModelForSW<VPU::ClampOp>>(*ctx);
        IE::ReLUOp::attachInterface<OpModelForSW<VPU::ReLUOp>>(*ctx);
        IE::EluOp::attachInterface<OpModelForSW<VPU::EluOp>>(*ctx);
        IE::HSwishOp::attachInterface<OpModelForSW<VPU::HSwishOp>>(*ctx);
        IE::ErfOp::attachInterface<OpModelForSW<VPU::ErfOp>>(*ctx);
        IE::MishOp::attachInterface<OpModelForSW<VPU::MishOp>>(*ctx);
        IE::FloorOp::attachInterface<OpModelForSW<VPU::FloorOp>>(*ctx);
        IE::RoundOp::attachInterface<OpModelForSW<VPU::RoundOp>>(*ctx);
        IE::TanhOp::attachInterface<OpModelForSW<VPU::TanhOp>>(*ctx);
        IE::SqrtOp::attachInterface<OpModelForSW<VPU::SqrtOp>>(*ctx);
        IE::SinhOp::attachInterface<OpModelForSW<VPU::SinhOp>>(*ctx);
        IE::CoshOp::attachInterface<OpModelForSW<VPU::CoshOp>>(*ctx);
        IE::AsinhOp::attachInterface<OpModelForSW<VPU::AsinhOp>>(*ctx);
        IE::AcoshOp::attachInterface<OpModelForSW<VPU::AcoshOp>>(*ctx);
        IE::AbsOp::attachInterface<OpModelForSW<VPU::AbsOp>>(*ctx);
        IE::AtanOp::attachInterface<OpModelForSW<VPU::AtanOp>>(*ctx);
        IE::AsinOp::attachInterface<OpModelForSW<VPU::AsinOp>>(*ctx);
        IE::AcosOp::attachInterface<OpModelForSW<VPU::AcosOp>>(*ctx);
        IE::AtanhOp::attachInterface<OpModelForSW<VPU::AtanhOp>>(*ctx);
        IE::GeluOp::attachInterface<OpModelForSW<VPU::GeluOp>>(*ctx);
        IE::ExpOp::attachInterface<OpModelForSW<VPU::ExpOp>>(*ctx);
        IE::ReduceMaxOp::attachInterface<OpModelForSW<VPU::ReduceMaxOp>>(*ctx);
        IE::ReduceMeanOp::attachInterface<OpModelForSW<VPU::ReduceMeanOp>>(*ctx);
        IE::ReduceLogicalOrOp::attachInterface<OpModelForSW<VPU::ReduceLogicalOrOp>>(*ctx);
        IE::ReduceLogicalAndOp::attachInterface<OpModelForSW<VPU::ReduceLogicalAndOp>>(*ctx);
        IE::ReduceProdOp::attachInterface<OpModelForSW<VPU::ReduceProdOp>>(*ctx);
        IE::ReduceSumOp::attachInterface<OpModelForSW<VPU::ReduceSumOp>>(*ctx);
        IE::ReduceMinOp::attachInterface<OpModelForSW<VPU::ReduceMinOp>>(*ctx);
        IE::ReduceL1Op::attachInterface<OpModelForSW<VPU::ReduceL1Op>>(*ctx);
        IE::ReduceL2Op::attachInterface<OpModelForSW<VPU::ReduceL2Op>>(*ctx);
        IE::LRN_IEOp::attachInterface<OpModelForSW<VPU::LRN_IEOp>>(*ctx);
        IE::UpsamplingOp::attachInterface<OpModelForSW<VPU::UpsamplingOp>>(*ctx);
        IE::ROIPoolingOp::attachInterface<OpModelForSW<VPU::ROIPoolingOp>>(*ctx);
        IE::PSROIPoolingOp::attachInterface<OpModelForSW<VPU::PSROIPoolingOp>>(*ctx);
        IE::CeilingOp::attachInterface<OpModelForSW<VPU::CeilingOp>>(*ctx);
        IE::DeformablePSROIPoolingOp::attachInterface<OpModelForSW<VPU::DeformablePSROIPoolingOp>>(*ctx);

        // VPU_SameInOutDimsOrder_NC_CHW_HWC_NCHW_NHWC
        IE::HSigmoidOp::attachInterface<OpModelForSW<VPU::HSigmoidOp>>(*ctx);

        // VPU_SameInOutDimsOrder_NCHW_NHWC
        IE::FakeQuantizeOp::attachInterface<OpModelForSW<VPU::FakeQuantizeOp>>(*ctx);
        IE::GRNOp::attachInterface<OpModelForSW<VPU::GRNOp>>(*ctx);

        // VPU_SameInOutDimsOrder_NCHW
        IE::ExtractImagePatchesOp::attachInterface<OpModelForSW<VPU::ExtractImagePatchesOp>>(*ctx);

        // VPU_AnyDimsOrder
        IE::DetectionOutputOp::attachInterface<OpModelForSW<VPU::DetectionOutputOp>>(*ctx);
        IE::EmbeddingBagOffsetsSumOp::attachInterface<OpModelForSW<VPU::EmbeddingBagOffsetsSumOp>>(*ctx);
        IE::EmbeddingSegmentsSumOp::attachInterface<OpModelForSW<VPU::EmbeddingSegmentsSumOp>>(*ctx);
        IE::EmbeddingBagPackedSumOp::attachInterface<OpModelForSW<VPU::EmbeddingBagPackedSumOp>>(*ctx);
        IE::BroadcastOp::attachInterface<OpModelForSW<VPU::BroadcastOp>>(*ctx);
        IE::ProposalOp::attachInterface<OpModelForSW<VPU::ProposalOp>>(*ctx);
        IE::ReorgYoloOp::attachInterface<OpModelForSW<VPU::ReorgYoloOp>>(*ctx);
        IE::GatherOp::attachInterface<OpModelForSW<VPU::GatherOp>>(*ctx);
        IE::ScatterNDUpdateOp::attachInterface<OpModelForSW<VPU::ScatterNDUpdateOp>>(*ctx);
        IE::ScatterUpdateOp::attachInterface<OpModelForSW<VPU::ScatterUpdateOp>>(*ctx);
        IE::YuvToRgbOp::attachInterface<OpModelForSW<VPU::YuvToRgbOp>>(*ctx);
        IE::LSTMSequenceOp::attachInterface<OpModelForSW<VPU::LSTMSequenceOp>>(*ctx);
        IE::GRUSequenceOp::attachInterface<OpModelForSW<VPU::GRUSequenceOp>>(*ctx);
        IE::GRUSequenceOp::attachInterface<OpModelForSW<VPU::GRUSequenceFirstPartOp>>(*ctx);
        IE::GRUSequenceOp::attachInterface<OpModelForSW<VPU::GRUSequenceLastPartOp>>(*ctx);
        IE::ReorderOp::attachInterface<OpModelForSW<VPU::MemPermuteOp>>(*ctx);
        IE::MemPermuteOp::attachInterface<OpModelForSW<VPU::MemPermuteOp>>(*ctx);
        IE::DynamicQuantizeOp::attachInterface<OpModelForSW<VPU::DynamicQuantizeOp>>(*ctx);

        // VPU_SameInOutDimsOrder_NCHW_CHW_NC_C
        IE::DivideOp::attachInterface<OpModelForSW<VPU::DivideOp>>(*ctx);
        IE::SquaredDifferenceOp::attachInterface<OpModelForSW<VPU::SquaredDifferenceOp>>(*ctx);
        IE::PowerOp::attachInterface<OpModelForSW<VPU::PowerOp>>(*ctx);
        IE::FloorModOp::attachInterface<OpModelForSW<VPU::FloorModOp>>(*ctx);
        IE::MinimumOp::attachInterface<OpModelForSW<VPU::MinimumOp>>(*ctx);
        IE::MaximumOp::attachInterface<OpModelForSW<VPU::MaximumOp>>(*ctx);
        IE::LogicalOrOp::attachInterface<OpModelForSW<VPU::LogicalOrOp>>(*ctx);
        IE::LogicalXorOp::attachInterface<OpModelForSW<VPU::LogicalXorOp>>(*ctx);
        IE::LessOp::attachInterface<OpModelForSW<VPU::LessOp>>(*ctx);
        IE::LessEqualOp::attachInterface<OpModelForSW<VPU::LessEqualOp>>(*ctx);
        IE::NotEqualOp::attachInterface<OpModelForSW<VPU::NotEqualOp>>(*ctx);
        IE::GreaterOp::attachInterface<OpModelForSW<VPU::GreaterOp>>(*ctx);
        IE::GreaterEqualOp::attachInterface<OpModelForSW<VPU::GreaterEqualOp>>(*ctx);
        IE::EqualOp::attachInterface<OpModelForSW<VPU::EqualOp>>(*ctx);
        IE::LogicalNotOp::attachInterface<OpModelForSW<VPU::LogicalNotOp>>(*ctx);
        IE::StridedSliceOp::attachInterface<OpModelForSW<VPU::StridedSliceOp>>(*ctx);

        // inferLayoutInfo
        IE::RandomUniformOp::attachInterface<OpModelForSW<VPU::RandomUniformOp>>(*ctx);
        IE::QuantizeOp::attachInterface<OpModelForSW<VPU::QuantizeOp>>(*ctx);
        IE::DequantizeOp::attachInterface<OpModelForSW<VPU::DequantizeOp>>(*ctx);
        IE::FullyConnectedOp::attachInterface<OpModelForSW<VPU::FullyConnectedOp>>(*ctx);
        IE::ScaleShiftOp::attachInterface<OpModelForSW<VPU::ScaleShiftOp>>(*ctx);
        IE::CTCGreedyDecoderOp::attachInterface<OpModelForSW<VPU::CTCGreedyDecoderOp>>(*ctx);
        IE::CTCGreedyDecoderSeqLenOp::attachInterface<OpModelForSW<VPU::CTCGreedyDecoderSeqLenOp>>(*ctx);
        IE::RegionYoloOp::attachInterface<OpModelForSW<VPU::RegionYoloOp>>(*ctx);
        IE::OneHotOp::attachInterface<OpModelForSW<VPU::OneHotOp>>(*ctx);
        IE::MVNOp::attachInterface<OpModelForSW<VPU::MVNOp>>(*ctx);
        IE::MVN6Op::attachInterface<OpModelForSW<VPU::MVN6Op>>(*ctx);
        IE::NonMaxSuppressionOp::attachInterface<OpModelForSW<VPU::NonMaxSuppressionOp>>(*ctx);
        IE::GatherNDOp::attachInterface<OpModelForSW<VPU::GatherNDOp>>(*ctx);
        IE::GatherTreeOp::attachInterface<OpModelForSW<VPU::GatherTreeOp>>(*ctx);
    });
}

}  // namespace

//
// setupExtraInterfaces
//

void vpux::VPU::VPUDialect::setupExtraInterfaces(mlir::DialectRegistry& registry) {
    redirectOpInterfacesForIE<LayoutInfoOpModelForSW>(registry);
}
