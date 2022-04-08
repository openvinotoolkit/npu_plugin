//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/ops.hpp"

#include "vpux/compiler/utils/asm.hpp"

using namespace vpux;

mlir::LogicalResult VPU::sameOrder(VPU::DistributedTensorType inDistributedType,
                                   VPU::DistributedTensorType outDistributedType, LogCb logCb) {
    if (inDistributedType.getOrder() != outDistributedType.getOrder()) {
        logCb(formatv("Mismatch between order for input ({0}) and output ({1}).", inDistributedType.getOrder(),
                      outDistributedType.getOrder()));
        return mlir::failure();
    }
    return mlir::success();
}

mlir::LogicalResult VPU::sameOrder(VPUIP::DistributedBufferType inDistributedType,
                                   VPUIP::DistributedBufferType outDistributedType, LogCb logCb) {
    if (inDistributedType.getLayout() != outDistributedType.getLayout()) {
        logCb(formatv("Mismatch between order for input ({0}) and output ({1}).", inDistributedType.getLayout(),
                      outDistributedType.getLayout()));
        return mlir::failure();
    }
    return mlir::success();
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
    // VPU_SameInOutDimsOrder
    registry.addOpInterface<IE::ConvertOp, OpModelForSW<VPU::ConvertOp>>();
    registry.addOpInterface<IE::SoftMaxOp, OpModelForSW<VPU::SoftMaxOp>>();
    registry.addOpInterface<IE::AvgPoolOp, OpModelForSW<VPU::AvgPoolOp>>();
    registry.addOpInterface<IE::AdaptiveAvgPoolOp, OpModelForSW<VPU::AdaptiveAvgPoolOp>>();
    registry.addOpInterface<IE::AdaptiveMaxPoolOp, OpModelForSW<VPU::AdaptiveMaxPoolOp>>();
    registry.addOpInterface<IE::RollOp, OpModelForSW<VPU::RollOp>>();
    registry.addOpInterface<IE::TanOp, OpModelForSW<VPU::TanOp>>();
    registry.addOpInterface<IE::SinOp, OpModelForSW<VPU::SinOp>>();
    registry.addOpInterface<IE::CosOp, OpModelForSW<VPU::CosOp>>();
    registry.addOpInterface<IE::LogOp, OpModelForSW<VPU::LogOp>>();
    registry.addOpInterface<IE::SeluOp, OpModelForSW<VPU::SeluOp>>();
    registry.addOpInterface<IE::PReluOp, OpModelForSW<VPU::PReluOp>>();
    registry.addOpInterface<IE::SelectOp, OpModelForSW<VPU::SelectOp>>();
    registry.addOpInterface<IE::SwishOp, OpModelForSW<VPU::SwishOp>>();
    registry.addOpInterface<IE::LeakyReluOp, OpModelForSW<VPU::LeakyReluOp>>();
    registry.addOpInterface<IE::HardSigmoidOp, OpModelForSW<VPU::HardSigmoidOp>>();
    registry.addOpInterface<IE::GridSampleOp, OpModelForSW<VPU::GridSampleOp>>();
    registry.addOpInterface<IE::BucketizeOp, OpModelForSW<VPU::BucketizeOp>>();
    registry.addOpInterface<IE::TileOp, OpModelForSW<VPU::TileOp>>();
    registry.addOpInterface<IE::PerAxisTileOp, OpModelForSW<VPU::PerAxisTileOp>>();
    registry.addOpInterface<IE::NegativeOp, OpModelForSW<VPU::NegativeOp>>();
    registry.addOpInterface<IE::PadOp, OpModelForSW<VPU::PadOp>>();
    registry.addOpInterface<IE::LSTMCellOp, OpModelForSW<VPU::LSTMCellOp>>();
    registry.addOpInterface<IE::SpaceToDepthOp, OpModelForSW<VPU::SpaceToDepthOp>>();
    registry.addOpInterface<IE::DepthToSpaceOp, OpModelForSW<VPU::DepthToSpaceOp>>();
    registry.addOpInterface<IE::NormalizeIEOp, OpModelForSW<VPU::NormalizeIEOp>>();
    registry.addOpInterface<IE::CumSumOp, OpModelForSW<VPU::CumSumOp>>();
    registry.addOpInterface<IE::ReverseSequenceOp, OpModelForSW<VPU::ReverseSequenceOp>>();
    registry.addOpInterface<IE::SoftPlusOp, OpModelForSW<VPU::SoftPlusOp>>();

    // VPU_SameInOutDimsOrder_CHW_HWC_NCHW_NHWC
    registry.addOpInterface<IE::SigmoidOp, OpModelForSW<VPU::SigmoidOp>>();
    registry.addOpInterface<IE::ClampOp, OpModelForSW<VPU::ClampOp>>();
    registry.addOpInterface<IE::ReLUOp, OpModelForSW<VPU::ReLUOp>>();
    registry.addOpInterface<IE::EluOp, OpModelForSW<VPU::EluOp>>();
    registry.addOpInterface<IE::HSwishOp, OpModelForSW<VPU::HSwishOp>>();
    registry.addOpInterface<IE::ErfOp, OpModelForSW<VPU::ErfOp>>();
    registry.addOpInterface<IE::MishOp, OpModelForSW<VPU::MishOp>>();
    registry.addOpInterface<IE::FloorOp, OpModelForSW<VPU::FloorOp>>();
    registry.addOpInterface<IE::RoundOp, OpModelForSW<VPU::RoundOp>>();
    registry.addOpInterface<IE::TanhOp, OpModelForSW<VPU::TanhOp>>();
    registry.addOpInterface<IE::SqrtOp, OpModelForSW<VPU::SqrtOp>>();
    registry.addOpInterface<IE::SinhOp, OpModelForSW<VPU::SinhOp>>();
    registry.addOpInterface<IE::CoshOp, OpModelForSW<VPU::CoshOp>>();
    registry.addOpInterface<IE::AsinhOp, OpModelForSW<VPU::AsinhOp>>();
    registry.addOpInterface<IE::AcoshOp, OpModelForSW<VPU::AcoshOp>>();
    registry.addOpInterface<IE::AbsOp, OpModelForSW<VPU::AbsOp>>();
    registry.addOpInterface<IE::HSigmoidOp, OpModelForSW<VPU::HSigmoidOp>>();
    registry.addOpInterface<IE::AtanOp, OpModelForSW<VPU::AtanOp>>();
    registry.addOpInterface<IE::AsinOp, OpModelForSW<VPU::AsinOp>>();
    registry.addOpInterface<IE::AcosOp, OpModelForSW<VPU::AcosOp>>();
    registry.addOpInterface<IE::AtanhOp, OpModelForSW<VPU::AtanhOp>>();
    registry.addOpInterface<IE::GeluOp, OpModelForSW<VPU::GeluOp>>();
    registry.addOpInterface<IE::ExpOp, OpModelForSW<VPU::ExpOp>>();
    registry.addOpInterface<IE::ReduceMaxOp, OpModelForSW<VPU::ReduceMaxOp>>();
    registry.addOpInterface<IE::ReduceMeanOp, OpModelForSW<VPU::ReduceMeanOp>>();
    registry.addOpInterface<IE::ReduceLogicalOrOp, OpModelForSW<VPU::ReduceLogicalOrOp>>();
    registry.addOpInterface<IE::ReduceLogicalAndOp, OpModelForSW<VPU::ReduceLogicalAndOp>>();
    registry.addOpInterface<IE::ReduceProdOp, OpModelForSW<VPU::ReduceProdOp>>();
    registry.addOpInterface<IE::ReduceSumOp, OpModelForSW<VPU::ReduceSumOp>>();
    registry.addOpInterface<IE::ReduceMinOp, OpModelForSW<VPU::ReduceMinOp>>();
    registry.addOpInterface<IE::ReduceL1Op, OpModelForSW<VPU::ReduceL1Op>>();
    registry.addOpInterface<IE::ReduceL2Op, OpModelForSW<VPU::ReduceL2Op>>();
    registry.addOpInterface<IE::LRN_IEOp, OpModelForSW<VPU::LRN_IEOp>>();
    registry.addOpInterface<IE::UpsamplingOp, OpModelForSW<VPU::UpsamplingOp>>();
    registry.addOpInterface<IE::ROIPoolingOp, OpModelForSW<VPU::ROIPoolingOp>>();
    registry.addOpInterface<IE::PSROIPoolingOp, OpModelForSW<VPU::PSROIPoolingOp>>();
    registry.addOpInterface<IE::CeilingOp, OpModelForSW<VPU::CeilingOp>>();
    registry.addOpInterface<IE::DeformablePSROIPoolingOp, OpModelForSW<VPU::DeformablePSROIPoolingOp>>();

    // VPU_SameInOutDimsOrder_NCHW_NHWC
    registry.addOpInterface<IE::FakeQuantizeOp, OpModelForSW<VPU::FakeQuantizeOp>>();
    registry.addOpInterface<IE::GRNOp, OpModelForSW<VPU::GRNOp>>();

    // VPU_SameInOutDimsOrder_NCHW
    registry.addOpInterface<IE::ExtractImagePatchesOp, OpModelForSW<VPU::ExtractImagePatchesOp>>();

    // VPU_AnyDimsOrder
    registry.addOpInterface<IE::DetectionOutputOp, OpModelForSW<VPU::DetectionOutputOp>>();
    registry.addOpInterface<IE::EmbeddingBagOffsetsSumOp, OpModelForSW<VPU::EmbeddingBagOffsetsSumOp>>();
    registry.addOpInterface<IE::EmbeddingSegmentsSumOp, OpModelForSW<VPU::EmbeddingSegmentsSumOp>>();
    registry.addOpInterface<IE::BroadcastOp, OpModelForSW<VPU::BroadcastOp>>();
    registry.addOpInterface<IE::ProposalOp, OpModelForSW<VPU::ProposalOp>>();
    registry.addOpInterface<IE::StridedSliceOp, OpModelForSW<VPU::StridedSliceOp>>();
    registry.addOpInterface<IE::ReorgYoloOp, OpModelForSW<VPU::ReorgYoloOp>>();
    registry.addOpInterface<IE::GatherOp, OpModelForSW<VPU::GatherOp>>();
    registry.addOpInterface<IE::ScatterNDUpdateOp, OpModelForSW<VPU::ScatterNDUpdateOp>>();
    registry.addOpInterface<IE::ScatterUpdateOp, OpModelForSW<VPU::ScatterUpdateOp>>();
    registry.addOpInterface<IE::YuvToRgbOp, OpModelForSW<VPU::YuvToRgbOp>>();
    registry.addOpInterface<IE::LSTMSequenceOp, OpModelForSW<VPU::LSTMSequenceOp>>();
    registry.addOpInterface<IE::GRUSequenceOp, OpModelForSW<VPU::GRUSequenceOp>>();
    registry.addOpInterface<IE::ReorderOp, OpModelForSW<VPU::MemPermuteOp>>();
    registry.addOpInterface<IE::MemPermuteOp, OpModelForSW<VPU::MemPermuteOp>>();

    // VPU_SameInOutDimsOrder_NCHW_CHW_NC_C
    registry.addOpInterface<IE::DivideOp, OpModelForSW<VPU::DivideOp>>();
    registry.addOpInterface<IE::SquaredDifferenceOp, OpModelForSW<VPU::SquaredDifferenceOp>>();
    registry.addOpInterface<IE::PowerOp, OpModelForSW<VPU::PowerOp>>();
    registry.addOpInterface<IE::FloorModOp, OpModelForSW<VPU::FloorModOp>>();
    registry.addOpInterface<IE::MinimumOp, OpModelForSW<VPU::MinimumOp>>();
    registry.addOpInterface<IE::MaximumOp, OpModelForSW<VPU::MaximumOp>>();
    registry.addOpInterface<IE::LogicalOrOp, OpModelForSW<VPU::LogicalOrOp>>();
    registry.addOpInterface<IE::LogicalXorOp, OpModelForSW<VPU::LogicalXorOp>>();
    registry.addOpInterface<IE::LessOp, OpModelForSW<VPU::LessOp>>();
    registry.addOpInterface<IE::LessEqualOp, OpModelForSW<VPU::LessEqualOp>>();
    registry.addOpInterface<IE::NotEqualOp, OpModelForSW<VPU::NotEqualOp>>();
    registry.addOpInterface<IE::GreaterOp, OpModelForSW<VPU::GreaterOp>>();
    registry.addOpInterface<IE::GreaterEqualOp, OpModelForSW<VPU::GreaterEqualOp>>();
    registry.addOpInterface<IE::EqualOp, OpModelForSW<VPU::EqualOp>>();
    registry.addOpInterface<IE::LogicalNotOp, OpModelForSW<VPU::LogicalNotOp>>();

    // inferLayoutInfo
    registry.addOpInterface<IE::QuantizeOp, OpModelForSW<VPU::QuantizeOp>>();
    registry.addOpInterface<IE::DequantizeOp, OpModelForSW<VPU::DequantizeOp>>();
    registry.addOpInterface<IE::FullyConnectedOp, OpModelForSW<VPU::FullyConnectedOp>>();
    registry.addOpInterface<IE::ScaleShiftOp, OpModelForSW<VPU::ScaleShiftOp>>();
    registry.addOpInterface<IE::CTCGreedyDecoderOp, OpModelForSW<VPU::CTCGreedyDecoderOp>>();
    registry.addOpInterface<IE::CTCGreedyDecoderSeqLenOp, OpModelForSW<VPU::CTCGreedyDecoderSeqLenOp>>();
    registry.addOpInterface<IE::InterpolateOp, OpModelForSW<VPU::InterpolateOp>>();
    registry.addOpInterface<IE::RegionYoloOp, OpModelForSW<VPU::RegionYoloOp>>();
    registry.addOpInterface<IE::MVNOp, OpModelForSW<VPU::MVNOp>>();
    registry.addOpInterface<IE::NonMaxSuppressionOp, OpModelForSW<VPU::NonMaxSuppressionOp>>();
    registry.addOpInterface<IE::GatherNDOp, OpModelForSW<VPU::GatherNDOp>>();
}

}  // namespace

//
// setupExtraInterfaces
//

void vpux::VPU::VPUDialect::setupExtraInterfaces(mlir::DialectRegistry& registry) {
    redirectOpInterfacesForIE<LayoutInfoOpModelForSW>(registry);
}
