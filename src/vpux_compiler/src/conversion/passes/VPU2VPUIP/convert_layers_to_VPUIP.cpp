//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/allocate_buffers.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

namespace {

//
// LayerRewrite
//

mlir::Operation* createRTLayer(VPU::QuantizeOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::QuantCastUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::QuantCastUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::DequantizeOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::QuantCastUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::QuantCastUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::ConvertOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ConvertUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ConvertUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::ReLUOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ReLUUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ReLUUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::SigmoidOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::SigmoidUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::SigmoidUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::SignOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::SignUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::SignUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::HSwishOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::HSwishUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::HSwishUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::FloorOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::FloorUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::FloorUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::RoundOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::RoundUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::RoundUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.mode());
}

mlir::Operation* createRTLayer(VPU::MishOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::MishUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::MishUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::ErfOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ErfUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ErfUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::TanOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::TanUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::TanUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::TanhOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::TanhUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::TanhUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::SinOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::SinUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::SinUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::CosOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::CosUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::CosUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::SqrtOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::SqrtUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::SqrtUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::SinhOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::SinhUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::SinhUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::CoshOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::CoshUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::CoshUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::AsinhOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::AsinhUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::AsinhUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::AcoshOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::AcoshUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::AcoshUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::AtanhOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::AtanhUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::AtanhUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::LogOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::LogUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::LogUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::GeluOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::GeluUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::GeluUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::NegativeOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::NegativeUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::NegativeUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::PReluOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::PReluUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::PReluUPAOp>(origOp.getLoc(), newOp.input(), newOp.negative_slope(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::GatherOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::GatherUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::GatherUPAOp>(origOp.getLoc(), newOp.input(), newOp.indices(), newOp.output_buff(),
                                        origOp.axis_valueAttr(), origOp.batch_dimsAttr());
}

mlir::Operation* createRTLayer(VPU::GatherNDOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::GatherNDUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::GatherNDUPAOp>(origOp.getLoc(), newOp.input(), newOp.indices(), newOp.output_buff(),
                                          origOp.batch_dimsAttr());
}

mlir::Operation* createRTLayer(VPU::YuvToRgbOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    const auto newInp2 = origOp.input2() != nullptr ? allBufs[2 - 1] : nullptr;
    const auto newInp3 = origOp.input3() != nullptr ? allBufs[3 - 1] : nullptr;
    return b.create<VPUIP::YuvToRgbUPAOp>(origOp.getLoc(), allBufs[0], newInp2, newInp3, allBufs.back(),
                                          origOp.inFmtAttr(), origOp.outFmtAttr());
}

mlir::Operation* createRTLayer(VPU::GatherElementsOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::GatherElementsUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::GatherElementsUPAOp>(origOp.getLoc(), newOp.input(), newOp.indices(), newOp.output_buff(),
                                                origOp.axisAttr());
}

mlir::Operation* createRTLayer(VPU::ScatterNDUpdateOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ScatterNDUpdateUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ScatterNDUpdateUPAOp>(origOp.getLoc(), newOp.input(), newOp.indices(), newOp.updates(),
                                                 newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::ScatterUpdateOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ScatterUpdateUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ScatterUpdateUPAOp>(origOp.getLoc(), newOp.input(), newOp.indices(), newOp.updates(),
                                               newOp.output_buff(), origOp.axis_valueAttr());
}

mlir::Operation* createRTLayer(VPU::AddOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::ADD));
}

mlir::Operation* createRTLayer(VPU::MultiplyOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::MULTIPLY));
}

mlir::Operation* createRTLayer(VPU::AndOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::AND));
}

mlir::Operation* createRTLayer(VPU::DivideOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::DIVIDE));
}

mlir::Operation* createRTLayer(VPU::SquaredDifferenceOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(
            origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff(),
            VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::SQUARED_DIFF));
}

mlir::Operation* createRTLayer(VPU::PowerOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::POWER));
}

mlir::Operation* createRTLayer(VPU::FloorModOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::FLOOR_MOD));
}

mlir::Operation* createRTLayer(VPU::MinimumOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::MIN));
}

mlir::Operation* createRTLayer(VPU::MaximumOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::MAX));
}

mlir::Operation* createRTLayer(VPU::SoftMaxOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::SoftMaxUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::SoftMaxUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.axisIndAttr());
}

mlir::Operation* createRTLayer(VPU::AvgPoolOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::PoolingUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::PoolingUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(),
                                         VPUIP::PoolLayerTypeAttr::get(origOp.getContext(), VPUIP::PoolLayerType::AVG),
                                         origOp.kernel_sizeAttr(), origOp.stridesAttr(), origOp.pads_beginAttr(),
                                         origOp.pads_endAttr(), origOp.exclude_padsAttr());
}

mlir::Operation* createRTLayer(VPU::MaxPoolOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::PoolingUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::PoolingUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(),
                                         VPUIP::PoolLayerTypeAttr::get(origOp.getContext(), VPUIP::PoolLayerType::MAX),
                                         origOp.kernel_sizeAttr(), origOp.stridesAttr(), origOp.pads_beginAttr(),
                                         origOp.pads_endAttr(), nullptr);
}

mlir::Operation* createRTLayer(VPU::ClampOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ClampUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ClampUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.minAttr(),
                                       origOp.maxAttr());
}

mlir::Operation* createRTLayer(VPU::EluOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EluUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EluUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.xAttr());
}

mlir::Operation* createRTLayer(VPU::LeakyReluOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::LeakyReluUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::LeakyReluUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(),
                                           origOp.negative_slopeAttr());
}

mlir::Operation* createRTLayer(VPU::GRNOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::GRNUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::GRNUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.biasAttr());
}

mlir::Operation* createRTLayer(VPU::LRN_IEOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::NormUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::NormUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.alphaAttr(),
                                      origOp.betaAttr(), origOp.biasAttr(), origOp.sizeAttr(), origOp.regionAttr());
}

mlir::Operation* createRTLayer(VPU::BroadcastOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::BroadcastUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::BroadcastUPAOp>(origOp.getLoc(), newOp.input(), newOp.target_shape(), newOp.axes_mapping(),
                                           newOp.output_buff(), origOp.modeAttr());
}

mlir::Operation* createRTLayer(VPU::ReduceMaxOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ReduceUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ReduceUPAOp>(
            origOp.getLoc(), newOp.input(), newOp.axes(), newOp.output_buff(), origOp.keep_dimsAttr(),
            VPUIP::ReduceLayerTypeAttr::get(origOp.getContext(), VPUIP::ReduceLayerType::MAX));
}

mlir::Operation* createRTLayer(VPU::ReduceMeanOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ReduceUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ReduceUPAOp>(
            origOp.getLoc(), newOp.input(), newOp.axes(), newOp.output_buff(), origOp.keep_dimsAttr(),
            VPUIP::ReduceLayerTypeAttr::get(origOp.getContext(), VPUIP::ReduceLayerType::MEAN));
}

mlir::Operation* createRTLayer(VPU::ReduceProdOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ReduceUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ReduceUPAOp>(
            origOp.getLoc(), newOp.input(), newOp.axes(), newOp.output_buff(), origOp.keep_dimsAttr(),
            VPUIP::ReduceLayerTypeAttr::get(origOp.getContext(), VPUIP::ReduceLayerType::PROD));
}

mlir::Operation* createRTLayer(VPU::ReduceSumOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ReduceUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ReduceUPAOp>(
            origOp.getLoc(), newOp.input(), newOp.axes(), newOp.output_buff(), origOp.keep_dimsAttr(),
            VPUIP::ReduceLayerTypeAttr::get(origOp.getContext(), VPUIP::ReduceLayerType::SUM));
}

mlir::Operation* createRTLayer(VPU::ReduceMinOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ReduceUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ReduceUPAOp>(
            origOp.getLoc(), newOp.input(), newOp.axes(), newOp.output_buff(), origOp.keep_dimsAttr(),
            VPUIP::ReduceLayerTypeAttr::get(origOp.getContext(), VPUIP::ReduceLayerType::MIN));
}

mlir::Operation* createRTLayer(VPU::ReduceL1Op origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ReduceUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ReduceUPAOp>(
            origOp.getLoc(), newOp.input(), newOp.axes(), newOp.output_buff(), origOp.keep_dimsAttr(),
            VPUIP::ReduceLayerTypeAttr::get(origOp.getContext(), VPUIP::ReduceLayerType::L1));
}

mlir::Operation* createRTLayer(VPU::ReduceL2Op origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ReduceUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ReduceUPAOp>(
            origOp.getLoc(), newOp.input(), newOp.axes(), newOp.output_buff(), origOp.keep_dimsAttr(),
            VPUIP::ReduceLayerTypeAttr::get(origOp.getContext(), VPUIP::ReduceLayerType::L2));
}

mlir::Operation* createRTLayer(VPU::ReduceLogicalOrOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ReduceUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ReduceUPAOp>(
            origOp.getLoc(), newOp.input(), newOp.axes(), newOp.output_buff(), origOp.keep_dimsAttr(),
            VPUIP::ReduceLayerTypeAttr::get(origOp.getContext(), VPUIP::ReduceLayerType::LOGICAL_OR));
}

mlir::Operation* createRTLayer(VPU::ReduceLogicalAndOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ReduceUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ReduceUPAOp>(
            origOp.getLoc(), newOp.input(), newOp.axes(), newOp.output_buff(), origOp.keep_dimsAttr(),
            VPUIP::ReduceLayerTypeAttr::get(origOp.getContext(), VPUIP::ReduceLayerType::LOGICAL_AND));
}

mlir::Operation* createRTLayer(VPU::PerAxisTileOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::PerAxisTileUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::PerAxisTileUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.axisAttr(),
                                             origOp.tilesAttr());
}

mlir::Operation* createRTLayer(VPU::ROIPoolingOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ROIPoolingUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ROIPoolingUPAOp>(origOp.getLoc(), newOp.input(), newOp.coords(), newOp.output_buff(),
                                            origOp.output_sizeAttr(), origOp.spatial_scaleAttr(), origOp.methodAttr());
}

mlir::Operation* createRTLayer(VPU::PSROIPoolingOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::PSROIPoolingUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::PSROIPoolingUPAOp>(origOp.getLoc(), newOp.input(), newOp.coords(), newOp.output_buff(),
                                              origOp.output_dimAttr(), origOp.spatial_scaleAttr(),
                                              origOp.group_sizeAttr(), origOp.spatial_bins_xAttr(),
                                              origOp.spatial_bins_yAttr(), origOp.modeAttr());
}

mlir::Operation* createRTLayer(VPU::ROIAlignOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ROIAlignUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ROIAlignUPAOp>(
            origOp.getLoc(), newOp.input(), newOp.coords(), newOp.roisIdx(), newOp.output_buff(), origOp.pooled_hAttr(),
            origOp.pooled_wAttr(), origOp.sampling_ratioAttr(), origOp.spatial_scaleAttr(), origOp.poolingModeAttr());
}

mlir::Operation* createRTLayer(VPU::GroupConvolutionOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ConvolutionUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ConvolutionUPAOp>(origOp.getLoc(), newOp.input(), newOp.filter(), newOp.bias(),
                                             newOp.output_buff(), origOp.stridesAttr(), origOp.dilationsAttr(),
                                             origOp.pads_beginAttr(), origOp.pads_endAttr(), origOp.groupsAttr());
}

mlir::Operation* createRTLayer(VPU::SwishOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::SwishUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::SwishUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.beta_valueAttr());
}

mlir::Operation* createRTLayer(VPU::DetectionOutputOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    const auto newPreds = origOp.in_additional_preds() != nullptr ? allBufs[3] : nullptr;
    const auto newProposals = origOp.in_additional_proposals() != nullptr ? allBufs[4] : nullptr;
    return b.create<VPUIP::DetectionOutputUPAOp>(origOp->getLoc(), allBufs[0], allBufs[1], allBufs[2], newPreds,
                                                 newProposals, allBufs.back(), origOp.attr());
}

mlir::Operation* createRTLayer(VPU::ScaleShiftOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    mlir::Value newWeights;
    mlir::Value newBiases;
    if (origOp.weights() != nullptr && origOp.biases() != nullptr) {
        newWeights = allBufs[1];
        newBiases = allBufs[2];
    } else if (origOp.weights() != nullptr) {
        newWeights = allBufs[1];
    } else if (origOp.biases() != nullptr) {
        newBiases = allBufs[1];
    } else {
        VPUX_THROW("ScaleShift must have weights or biases");
    }
    return b.create<VPUIP::ScaleShiftUPAOp>(origOp->getLoc(), allBufs[0], newWeights, newBiases, allBufs.back());
}

mlir::Operation* createRTLayer(VPU::CTCGreedyDecoderOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::CTCGreedyDecoderUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::CTCGreedyDecoderUPAOp>(origOp.getLoc(), newOp.input(), newOp.sequenceLengths(),
                                                  newOp.output_buff(), origOp.mergeRepeatedAttr());
}

mlir::Operation* createRTLayer(VPU::CTCGreedyDecoderSeqLenOp origOp, ArrayRef<mlir::Value> allBufs,
                               mlir::OpBuilder& b) {
    VPUIP::CTCGreedyDecoderSeqLenUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::CTCGreedyDecoderSeqLenUPAOp>(origOp.getLoc(), newOp.input(), newOp.sequenceLength(),
                                                        newOp.blankIndex(), newOp.output_buff(),
                                                        newOp.outputLength_buff(), origOp.mergeRepeatedAttr());
}

mlir::Operation* createRTLayer(VPU::ProposalOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ProposalUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ProposalUPAOp>(origOp.getLoc(), newOp.class_probs(), newOp.bbox_deltas(),
                                          newOp.image_shape(), newOp.output_buff(), newOp.probs_buff(),
                                          origOp.proposal_attrs());
}

mlir::Operation* createRTLayer(VPU::RollOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::RollUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::RollUPAOp>(origOp.getLoc(), newOp.data(), newOp.shift(), newOp.axes(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::PadOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUX_THROW_UNLESS(origOp.pads_begin_attr().hasValue() && origOp.pads_end_attr().hasValue(),
                      "PadOp must use attributes for `pads_begin` and `pads_end` params");

    VPUIP::PadUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::PadUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.pads_begin_attrAttr(),
                                     origOp.pads_end_attrAttr(), origOp.pad_value_attrAttr(), origOp.modeAttr());
}

mlir::Operation* createRTLayer(VPU::ExpOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ExpUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ExpUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::InterpolateOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUX_THROW_UNLESS(origOp.sizes_attr().hasValue() && origOp.scales_attr().hasValue(),
                      "Interpolate must have constant sizes or scales");
    VPUX_THROW_UNLESS(origOp.axes_attr().hasValue(), "Interpolate must have constant axes");

    VPUIP::InterpolateUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::InterpolateUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(),
                                             origOp.attr().mode().getValue(), origOp.attr().coord_mode().getValue(),
                                             origOp.attr().nearest_mode().getValue(),
                                             origOp.attr().antialias().getValue());
}

mlir::Operation* createRTLayer(VPU::StridedSliceOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUX_THROW_UNLESS(allBufs.size() == 2, "Constant inputs should have been converted to attributes");
    VPUX_THROW_UNLESS(origOp.begins_attrAttr(), "begins_attr is null");
    VPUX_THROW_UNLESS(origOp.ends_attrAttr(), "ends_attr is null");
    VPUX_THROW_UNLESS(origOp.strides_attrAttr(), "strides_attr is null");

    return b.create<VPUIP::StridedSliceUPAOp>(origOp.getLoc(), allBufs[0], allBufs.back(), origOp.begins_attrAttr(),
                                              origOp.ends_attrAttr(), origOp.strides_attrAttr());
}

mlir::Operation* createRTLayer(VPU::RegionYoloOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::RegionYoloUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::RegionYoloUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.coords(),
                                            origOp.classes(), origOp.regions(), origOp.do_softmaxAttr(), origOp.mask());
}

mlir::Operation* createRTLayer(VPU::ReorgYoloOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ReorgYoloUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ReorgYoloUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.strideAttr());
}

mlir::Operation* createRTLayer(VPU::MVNOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::MVNUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::MVNUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.across_channelsAttr(),
                                     origOp.normalize_varianceAttr(), origOp.epsAttr());
}

mlir::Operation* createRTLayer(VPU::DepthToSpaceOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::DepthToSpaceUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::DepthToSpaceUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(),
                                              origOp.block_sizeAttr(), origOp.modeAttr());
}

mlir::Operation* createRTLayer(VPU::MemPermuteOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::PermuteUPAOp::Adaptor newOp(allBufs);

    return b.create<VPUIP::PermuteUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.mem_perm());
}

mlir::Operation* createRTLayer(VPU::SoftPlusOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::SoftPlusUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::SoftPlusUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::CeilingOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::CeilingUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::CeilingUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::NormalizeIEOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::NormalizeIEUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::NormalizeIEUPAOp>(origOp.getLoc(), newOp.data(), newOp.weights(), newOp.output_buff(),
                                             origOp.eps(), origOp.across_spatial(), origOp.channel_shared());
}

mlir::Operation* createRTLayer(VPU::CumSumOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::CumSumUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::CumSumUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.axis_valueAttr(),
                                        origOp.exclusiveAttr(), origOp.reverseAttr());
}

mlir::Operation* createRTLayer(VPU::EqualOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::EQUAL));
}

mlir::Operation* createRTLayer(VPU::SelectOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::SelectUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::SelectUPAOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.input3(),
                                        newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::UpsamplingOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::UpsamplingUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::UpsamplingUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(),
                                            origOp.upsampling_factorAttr(), origOp.pad_lAttr(), origOp.pad_rAttr());
}

mlir::Operation* createRTLayer(VPU::LessOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::LESS));
}

mlir::Operation* createRTLayer(VPU::LessEqualOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::LESS_EQUAL));
}

mlir::Operation* createRTLayer(VPU::NotEqualOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::NOT_EQUAL));
}

mlir::Operation* createRTLayer(VPU::GreaterOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::GREATER));
}

mlir::Operation* createRTLayer(VPU::GreaterEqualOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(
            origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff(),
            VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::GREATER_EQUAL));
}

mlir::Operation* createRTLayer(VPU::LogicalNotOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::LogicalNotUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::LogicalNotUPAOp>(
            origOp.getLoc(), newOp.input1(), newOp.output_buff(),
            VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::LOGICAL_NOT));
}

mlir::Operation* createRTLayer(VPU::LogicalOrOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::LOGICAL_OR));
}

mlir::Operation* createRTLayer(VPU::LogicalXorOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EltwiseUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EltwiseUPAOp>(origOp.getLoc(), newOp.input1(), newOp.input2(), newOp.output_buff(),
                                         VPU::EltwiseTypeAttr::get(origOp.getContext(), VPU::EltwiseType::LOGICAL_XOR));
}

mlir::Operation* createRTLayer(VPU::SpaceToDepthOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::SpaceToDepthUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::SpaceToDepthUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.block_size(),
                                              origOp.mode());
}

mlir::Operation* createRTLayer(VPU::CopyOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::CopyOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::CopyOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::ExpandOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ExpandOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ExpandOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(), origOp.pads_begin(),
                                     origOp.pads_end());
}

mlir::Operation* createRTLayer(VPU::AbsOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::AbsUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::AbsUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::AtanOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::AtanUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::AtanUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::AsinOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::AsinUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::AsinUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::AcosOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::AcosUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::AcosUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::HSigmoidOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::HSigmoidUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::HSigmoidUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::HardSigmoidOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::HardSigmoidUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::HardSigmoidUPAOp>(origOp.getLoc(), newOp.input(), newOp.output_buff(),
                                             origOp.alpha_valueAttr(), origOp.beta_valueAttr());
}

mlir::Operation* createRTLayer(VPU::EmbeddingSegmentsSumOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::EmbeddingSegmentsSumUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EmbeddingSegmentsSumUPAOp>(origOp.getLoc(), newOp.emb_table(), newOp.output_buff(),
                                                      origOp.indices_valueAttr(), origOp.segment_ids_valueAttr(),
                                                      origOp.num_segments_valueAttr(), origOp.default_index_valueAttr(),
                                                      origOp.per_sample_weights_valueAttr());
}

mlir::Operation* createRTLayer(VPU::BucketizeOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::BucketizeUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::BucketizeUPAOp>(origOp.getLoc(), newOp.data(), newOp.buckets(), newOp.output_buff(),
                                           origOp.output_typeAttr(), origOp.with_right_boundAttr());
}

mlir::Operation* createRTLayer(VPU::ExtractImagePatchesOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::ExtractImagePatchesUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::ExtractImagePatchesUPAOp>(origOp.getLoc(), newOp.data(), newOp.output_buff(),
                                                     origOp.sizesAttr(), origOp.stridesAttr(), origOp.ratesAttr(),
                                                     origOp.autoPadAttr());
}

mlir::Operation* createRTLayer(VPU::AdaptiveAvgPoolOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::AdaptiveAvgPoolUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::AdaptiveAvgPoolUPAOp>(origOp.getLoc(), newOp.input(), newOp.pooled_spatial_shape(),
                                                 newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::AdaptiveMaxPoolOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::AdaptiveMaxPoolUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::AdaptiveMaxPoolUPAOp>(origOp.getLoc(), newOp.input(), newOp.pooled_spatial_shape(),
                                                 newOp.output_buff(), newOp.output_index_buff(),
                                                 origOp.index_element_typeAttr());
}

mlir::Operation* createRTLayer(VPU::SeluOp origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    VPUIP::SeluUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::SeluUPAOp>(origOp.getLoc(), newOp.data(), newOp.output_buff(), origOp.alpha_valueAttr(),
                                      origOp.lambda_valueAttr());
}

mlir::Operation* createRTLayer(VPU::EmbeddingBagOffsetsSumOp origOp, ArrayRef<mlir::Value> allBufs,
                               mlir::OpBuilder& b) {
    VPUIP::EmbeddingBagOffsetsSumUPAOp::Adaptor newOp(allBufs);
    return b.create<VPUIP::EmbeddingBagOffsetsSumUPAOp>(origOp.getLoc(), newOp.input(), origOp.indices_valueAttr(),
                                                        origOp.offsets_valueAttr(), origOp.default_index_valueAttr(),
                                                        origOp.weights_valueAttr(), newOp.output_buff());
}

mlir::Operation* createRTLayer(VPU::DeformablePSROIPoolingOp origOp, ArrayRef<mlir::Value> allBufs,
                               mlir::OpBuilder& b) {
    VPUIP::DeformablePSROIPoolingUPAOp::Adaptor newOp(allBufs);
    const auto newInp3 = origOp.input_transformations() != nullptr ? allBufs[2] : nullptr;
    return b.create<VPUIP::DeformablePSROIPoolingUPAOp>(
            origOp.getLoc(), newOp.input_score_maps(), newOp.input_rois(), newInp3, newOp.output_buff(),
            origOp.output_dimAttr(), origOp.spatial_scaleAttr(), origOp.group_sizeAttr(), origOp.spatial_bins_xAttr(),
            origOp.spatial_bins_yAttr(), origOp.trans_stdAttr(), origOp.part_sizeAttr(), origOp.modeAttr());
}

class LayerRewrite final : public mlir::ConversionPattern {
public:
    LayerRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::ConversionPattern(typeConverter, mlir::Pattern::MatchAnyOpTypeTag{}, benefitLow, ctx), _log(log) {
        setDebugName("LayerRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::Operation* origOp, ArrayRef<mlir::Value> newOperands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    template <class InLayerOp>
    static mlir::Operation* dispatch(mlir::Operation* origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b);

private:
    Logger _log;
};

template <class InLayerOp>
mlir::Operation* LayerRewrite::dispatch(mlir::Operation* origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder& b) {
    return createRTLayer(mlir::cast<InLayerOp>(origOp), allBufs, b);
}

mlir::LogicalResult LayerRewrite::matchAndRewrite(mlir::Operation* origOp, ArrayRef<mlir::Value> newOperands,
                                                  mlir::ConversionPatternRewriter& rewriter) const {
    using CreateFunc =
            mlir::Operation* (*)(mlir::Operation * origOp, ArrayRef<mlir::Value> allBufs, mlir::OpBuilder & b);

#define CASE(_OP_)                                   \
    .Case<_OP_>([](mlir::Operation*) -> CreateFunc { \
        return dispatch<_OP_>;                       \
    })

    const CreateFunc createFunc = llvm::TypeSwitch<mlir::Operation*, CreateFunc>(origOp)  //
            CASE(VPU::QuantizeOp)
    CASE(VPU::DequantizeOp)
    CASE(VPU::ConvertOp)
    CASE(VPU::SoftMaxOp)
    CASE(VPU::AvgPoolOp)
    CASE(VPU::MaxPoolOp)
    CASE(VPU::GroupConvolutionOp)
    CASE(VPU::ReLUOp)
    CASE(VPU::SigmoidOp)
    CASE(VPU::SignOp)
    CASE(VPU::ClampOp)
    CASE(VPU::EluOp)
    CASE(VPU::HSwishOp)
    CASE(VPU::FloorOp)
    CASE(VPU::RoundOp)
    CASE(VPU::MishOp)
    CASE(VPU::ErfOp)
    CASE(VPU::BroadcastOp)
    CASE(VPU::ReduceMaxOp)
    CASE(VPU::ReduceMeanOp)
    CASE(VPU::ReduceProdOp)
    CASE(VPU::ReduceSumOp)
    CASE(VPU::ReduceMinOp)
    CASE(VPU::ReduceL1Op)
    CASE(VPU::ReduceL2Op)
    CASE(VPU::ReduceLogicalOrOp)
    CASE(VPU::ReduceLogicalAndOp)
    CASE(VPU::TanOp)
    CASE(VPU::TanhOp)
    CASE(VPU::SinOp)
    CASE(VPU::CosOp)
    CASE(VPU::SqrtOp)
    CASE(VPU::SinhOp)
    CASE(VPU::CoshOp)
    CASE(VPU::AsinhOp)
    CASE(VPU::AcoshOp)
    CASE(VPU::AtanhOp)
    CASE(VPU::LogOp)
    CASE(VPU::GeluOp)
    CASE(VPU::PReluOp)
    CASE(VPU::GatherOp)
    CASE(VPU::GatherNDOp)
    CASE(VPU::YuvToRgbOp)
    CASE(VPU::GatherElementsOp)
    CASE(VPU::ScatterNDUpdateOp)
    CASE(VPU::ScatterUpdateOp)
    CASE(VPU::LeakyReluOp)
    CASE(VPU::AddOp)
    CASE(VPU::MultiplyOp)
    CASE(VPU::AndOp)
    CASE(VPU::DivideOp)
    CASE(VPU::SquaredDifferenceOp)
    CASE(VPU::PowerOp)
    CASE(VPU::FloorModOp)
    CASE(VPU::MinimumOp)
    CASE(VPU::MaximumOp)
    CASE(VPU::SwishOp)
    CASE(VPU::GRNOp)
    CASE(VPU::LRN_IEOp)
    CASE(VPU::PerAxisTileOp)
    CASE(VPU::NegativeOp)
    CASE(VPU::ROIPoolingOp)
    CASE(VPU::PSROIPoolingOp)
    CASE(VPU::ROIAlignOp)
    CASE(VPU::DetectionOutputOp)
    CASE(VPU::ScaleShiftOp)
    CASE(VPU::CTCGreedyDecoderOp)
    CASE(VPU::CTCGreedyDecoderSeqLenOp)
    CASE(VPU::ProposalOp)
    CASE(VPU::RollOp)
    CASE(VPU::PadOp)
    CASE(VPU::ExpOp)
    CASE(VPU::InterpolateOp)
    CASE(VPU::StridedSliceOp)
    CASE(VPU::RegionYoloOp)
    CASE(VPU::ReorgYoloOp)
    CASE(VPU::MVNOp)
    CASE(VPU::DepthToSpaceOp)
    CASE(VPU::MemPermuteOp)
    CASE(VPU::SoftPlusOp)
    CASE(VPU::CeilingOp)
    CASE(VPU::NormalizeIEOp)
    CASE(VPU::CumSumOp)
    CASE(VPU::EqualOp)
    CASE(VPU::SelectOp)
    CASE(VPU::UpsamplingOp)
    CASE(VPU::LessOp)
    CASE(VPU::LessEqualOp)
    CASE(VPU::NotEqualOp)
    CASE(VPU::GreaterOp)
    CASE(VPU::GreaterEqualOp)
    CASE(VPU::SpaceToDepthOp)
    CASE(VPU::LogicalNotOp)
    CASE(VPU::LogicalOrOp)
    CASE(VPU::LogicalXorOp)
    CASE(VPU::CopyOp)
    CASE(VPU::ExpandOp)
    CASE(VPU::AbsOp)
    CASE(VPU::AtanOp)
    CASE(VPU::AsinOp)
    CASE(VPU::AcosOp)
    CASE(VPU::HSigmoidOp)
    CASE(VPU::HardSigmoidOp)
    CASE(VPU::BucketizeOp)
    CASE(VPU::ExtractImagePatchesOp)
    CASE(VPU::AdaptiveAvgPoolOp)
    CASE(VPU::AdaptiveMaxPoolOp)
    CASE(VPU::SeluOp)
    CASE(VPU::EmbeddingBagOffsetsSumOp)
    CASE(VPU::EmbeddingSegmentsSumOp)
    CASE(VPU::DeformablePSROIPoolingOp)
    .Default([](mlir::Operation*) {
        return nullptr;
    });

#undef CASE

    if (createFunc == nullptr) {
        return mlir::failure();
    }

    _log.trace("Found Layer Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    VPUX_THROW_UNLESS(newOperands.size() == origOp->getNumOperands(), "Got wrong newOperands size : '{0}'",
                      newOperands.size());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    const auto resultBufs = allocateBuffers(_log, origOp->getLoc(), rewriter, *typeConverter, origOp->getOpResults());

    SmallVector<mlir::Value> allBufs;
    allBufs.reserve(newOperands.size() + resultBufs.size());
    allBufs.append(newOperands.begin(), newOperands.end());
    allBufs.append(resultBufs.begin(), resultBufs.end());

    const auto newOp = createFunc(origOp, allBufs, rewriter);
    rewriter.replaceOp(origOp, newOp->getResults());

    return mlir::success();
}

//
// ReshapeRewrite
//

template <class ConcreteOp>
class ReshapeRewrite final : public mlir::OpConversionPattern<ConcreteOp> {
    using OpAdaptor = typename mlir::OpConversionPattern<ConcreteOp>::OpAdaptor;

public:
    ReshapeRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<ConcreteOp>(typeConverter, ctx), _log(log) {
        this->setDebugName("ReshapeRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult ReshapeRewrite<ConcreteOp>::matchAndRewrite(ConcreteOp origOp, OpAdaptor newArgs,
                                                                mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found Reshape Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto* typeConverter = this->getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    const auto newOutType = typeConverter->convertType(origOp.getType());

    rewriter.replaceOpWithNewOp<VPUIP::GenericReshapeOp>(origOp, newOutType, newArgs.input());
    return mlir::success();
}

mlir::OpResult createCopyResult(mlir::Type type, mlir::Value inputBuffer, mlir::Value outputBuffer,
                                mlir::ConversionPatternRewriter& rewriter, mlir::Location location) {
    if (type == nullptr) {
        return mlir::OpResult();
    }

    auto dataType = type;
    if (auto sparseBuffer = dataType.dyn_cast<VPUIP::SparseBufferType>()) {
        dataType = sparseBuffer.getData();
    }

    if (dataType.isa<mlir::MemRefType, VPUIP::BufferType>()) {
        auto copyOp = rewriter.create<VPUIP::CopyOp>(location, inputBuffer, outputBuffer);

        return copyOp.getOperation()->getResult(0);
    } else if (dataType.isa<VPUIP::DistributedBufferType>()) {
        // Create NCEClusterTiling with CopyOp inside
        SmallVector<mlir::Value> inputsOutputOperands = {inputBuffer, outputBuffer};

        const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
            builder.create<VPUIP::CopyOp>(loc, newOperands[0], newOperands[1]);
        };

        auto clusterTilingOp =
                rewriter.create<VPUIP::NCEClusterTilingOp>(location, type, inputsOutputOperands, bodyBuilder);
        return clusterTilingOp.getOperation()->getResult(0);
    }
    VPUX_THROW("Unexpected data type to copy: {0}", dataType);
};

//
// SliceRewrite
//

class SliceRewrite final : public mlir::OpConversionPattern<VPU::SliceOp> {
    using OpAdaptor = typename mlir::OpConversionPattern<VPU::SliceOp>::OpAdaptor;

public:
    SliceRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::SliceOp>(typeConverter, ctx), _log(log) {
        this->setDebugName("SliceRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::SliceOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SliceRewrite::matchAndRewrite(VPU::SliceOp origOp, OpAdaptor newArgs,
                                                  mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found VPU::Slice Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    const auto origType = origOp.getType();
    const auto newOutType = typeConverter->convertType(origType);

    auto subView = rewriter.create<VPUIP::SubViewOp>(origOp->getLoc(), newArgs.source(), origOp.static_offsetsAttr(),
                                                     origOp.static_sizesAttr());

    auto outputBuffers = allocateBuffers(_log, origOp->getLoc(), rewriter, *typeConverter, origOp->getOpResults());

    mlir::OpResult newResult =
            createCopyResult(newOutType, subView.result(), outputBuffers[0], rewriter, origOp->getLoc());

    rewriter.replaceOp(origOp, newResult);

    return mlir::success();
}

//
// SplitRewrite
//

class SplitRewrite final : public mlir::OpConversionPattern<VPU::SplitOp> {
public:
    SplitRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::SplitOp>(typeConverter, ctx), _log(log) {
        setDebugName("SplitRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::SplitOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SplitRewrite::matchAndRewrite(VPU::SplitOp origOp, OpAdaptor newArgs,
                                                  mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found Split Operation '{0}'", origOp->getLoc());

    if (!origOp.axis_value().hasValue()) {
        return matchFailed(rewriter, origOp, "Got non constant axis");
    }

    const auto inputType = newArgs.input().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inputType.getShape();

    const auto axis = Dim(origOp.axis_value().getValue());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    auto allocatedBufs = allocateBuffers(_log, origOp->getLoc(), rewriter, *typeConverter, origOp.getResults());

    // Prepare strides array for subview. We have dense array, so all strides have to be equal 1
    SmallVector<int64_t> svOffsets(inputShape.size(), 0);
    SmallVector<mlir::Value> results;

    const auto offsetStep = inputShape[axis] / origOp.num_splits();

    for (auto i : irange(origOp->getNumResults())) {
        const auto origOutputType = origOp.getResult(i).getType().cast<vpux::NDTypeInterface>();
        const auto svSizes = origOutputType.getShape().raw();

        _log.trace("Create SubView for output #'{0}'", i);
        auto subView = rewriter.create<VPUIP::SubViewOp>(origOp.getLoc(), newArgs.input(), svOffsets, svSizes);

        _log.trace("Copy SubView result to output buffer");

        auto copyOp = rewriter.create<VPUIP::CopyOp>(origOp->getLoc(), subView, allocatedBufs[i]);
        results.push_back(copyOp.output());

        svOffsets[axis.ind()] += offsetStep;
    }

    rewriter.replaceOp(origOp, results);

    return mlir::success();
}

//
// ConcatRewrite
//

class ConcatRewrite final : public mlir::OpConversionPattern<VPU::ConcatOp> {
public:
    ConcatRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::ConcatOp>(typeConverter, ctx), _log(log) {
        setDebugName("ConcatRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::ConcatOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    SmallVector<mlir::Value> rewriteWithAxis(VPU::ConcatOp origOp, OpAdaptor newArgs,
                                             ArrayRef<mlir::Value> allocatedBufs,
                                             mlir::ConversionPatternRewriter& rewriter) const;
    SmallVector<mlir::Value> rewriteWithOffsets(VPU::ConcatOp origOp, OpAdaptor newArgs,
                                                ArrayRef<mlir::Value> allocatedBufs,
                                                mlir::ConversionPatternRewriter& rewriter) const;

private:
    Logger _log;
};

SmallVector<mlir::Value> ConcatRewrite::rewriteWithAxis(VPU::ConcatOp origOp, OpAdaptor newArgs,
                                                        ArrayRef<mlir::Value> allocatedBufs,
                                                        mlir::ConversionPatternRewriter& rewriter) const {
    SmallVector<mlir::Value> results;

    const auto axis = origOp.per_axisAttr().axis().getValue().getSExtValue();
    const auto offset = origOp.per_axisAttr().offset() ? origOp.per_axisAttr().offset().getValue().getSExtValue() : 0;
    const auto stride = origOp.per_axisAttr().stride() ? origOp.per_axisAttr().stride().getValue().getSExtValue() : 1;

    const auto outputRank = origOp.getType().cast<vpux::NDTypeInterface>().getRank();

    SmallVector<int64_t> svOffsets(outputRank, 0);

    SmallVector<int64_t> svElemStrides;
    if (stride != 1) {
        svElemStrides.resize(outputRank, 1);
        svElemStrides[axis] = stride;
    }

    for (auto i : irange(origOp->getNumOperands())) {
        const auto newInput = newArgs.inputs()[i];
        const auto newInputType = newInput.getType().cast<vpux::NDTypeInterface>();
        const auto svSizes = newInputType.getShape().raw();

        _log.trace("Create SubView for input #'{0}'", i);
        mlir::Value subViewVal;
        if (svElemStrides.empty()) {
            subViewVal = rewriter.create<VPUIP::SubViewOp>(origOp->getLoc(), allocatedBufs[0], svOffsets, svSizes);
            svOffsets[axis] += svSizes[axis];
        } else {
            subViewVal = rewriter.create<VPUIP::SubViewOp>(origOp->getLoc(), allocatedBufs[0], svOffsets, svSizes,
                                                           svElemStrides);
            svOffsets[axis] += offset;
        }

        _log.trace("Copy new operand to SubView");

        auto newOutType = subViewVal.getType();

        // Copy to the SubView
        mlir::OpResult newResult = createCopyResult(newOutType, newInput, subViewVal, rewriter, origOp->getLoc());
        results.push_back(newResult);
    }

    return results;
}

SmallVector<mlir::Value> ConcatRewrite::rewriteWithOffsets(VPU::ConcatOp origOp, OpAdaptor newArgs,
                                                           ArrayRef<mlir::Value> allocatedBufs,
                                                           mlir::ConversionPatternRewriter& rewriter) const {
    SmallVector<mlir::Value> results;

    const auto allOffsets = origOp.static_offsetsAttr().getAsRange<mlir::ArrayAttr>();

    for (const auto p : zip(newArgs.inputs(), allOffsets)) {
        const auto newInput = std::get<0>(p);

        const auto curShape = newInput.getType().cast<vpux::NDTypeInterface>().getShape().raw();
        const auto curOffsets = parseIntArrayAttr<int64_t>(std::get<1>(p));

        _log.trace("Create SubView");

        auto subViewOp = rewriter.create<VPUIP::SubViewOp>(origOp->getLoc(), allocatedBufs[0], curOffsets, curShape);

        _log.trace("Copy new operand to SubView");

        auto newOutType = subViewOp.result().getType();

        // Copy to the SubView
        mlir::OpResult newResult =
                createCopyResult(newOutType, newInput, subViewOp.result(), rewriter, origOp->getLoc());
        results.push_back(newResult);
    }

    return results;
}

mlir::LogicalResult ConcatRewrite::matchAndRewrite(VPU::ConcatOp origOp, OpAdaptor newArgs,
                                                   mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found Concat Operation '{0}'", origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    _log.trace("Add Alloc Operations for results");

    auto newOutType = typeConverter->convertType(origOp.getResult().getType());

    SmallVector<mlir::Value> allocatedBufs;

    auto outputBuffers = allocateBuffers(_log, origOp->getLoc(), rewriter, *typeConverter, origOp->getOpResults());

    const auto results = origOp.per_axisAttr() ? rewriteWithAxis(origOp, newArgs, outputBuffers, rewriter)
                                               : rewriteWithOffsets(origOp, newArgs, outputBuffers, rewriter);

    rewriter.replaceOpWithNewOp<VPUIP::ConcatViewOp>(origOp, newOutType, results, outputBuffers[0]);
    return mlir::success();
}

//
// PermuteCastRewrite
//

class PermuteCastRewrite final : public mlir::OpConversionPattern<VPU::PermuteCastOp> {
public:
    PermuteCastRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::PermuteCastOp>(typeConverter, ctx), _log(log) {
        setDebugName("PermuteCastRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::PermuteCastOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult PermuteCastRewrite::matchAndRewrite(VPU::PermuteCastOp origOp, OpAdaptor newArgs,
                                                        mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found PermuteCast Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    const auto newOutType = typeConverter->convertType(origOp.getType());

    rewriter.replaceOpWithNewOp<VPUIP::PermuteCastOp>(origOp, newOutType, newArgs.input(), origOp.dst_orderAttr(),
                                                      origOp.mem_permAttr());
    return mlir::success();
}

//
// QuantizeCastRewriter
//

class QuantizeCastRewriter final : public mlir::OpConversionPattern<VPU::QuantizeCastOp> {
    using OpAdaptor = typename mlir::OpConversionPattern<VPU::QuantizeCastOp>::OpAdaptor;

public:
    QuantizeCastRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::QuantizeCastOp>(typeConverter, ctx), _log(log) {
        this->setDebugName("QuantizeCastRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::QuantizeCastOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult QuantizeCastRewriter::matchAndRewrite(VPU::QuantizeCastOp origOp, OpAdaptor newArgs,
                                                          mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found QuantizeCast Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    const auto outType = origOp.getType();

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    const auto newOutType = typeConverter->convertType(outType);

    rewriter.replaceOpWithNewOp<VPUIP::QuantizeCastOp>(origOp, newOutType, newArgs.input());
    return mlir::success();
}

//
// DistributedCastRewriter
//

class DistributedCastRewriter final : public mlir::OpConversionPattern<VPU::DistributedCastOp> {
    using OpAdaptor = typename mlir::OpConversionPattern<VPU::DistributedCastOp>::OpAdaptor;

public:
    DistributedCastRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::DistributedCastOp>(typeConverter, ctx), _log(log) {
        this->setDebugName("DistributedCastRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::DistributedCastOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DistributedCastRewriter::matchAndRewrite(VPU::DistributedCastOp origOp, OpAdaptor newArgs,
                                                             mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found DistributedCast Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    const auto newOutType = typeConverter->convertType(origOp.getType());

    rewriter.replaceOpWithNewOp<VPUIP::DistributedCastOp>(origOp, newOutType, newArgs.input());
    return mlir::success();
}

//
// ReverseSequenceRewrite
//

class ReverseSequenceRewrite final : public mlir::OpConversionPattern<VPU::ReverseSequenceOp> {
public:
    ReverseSequenceRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::ReverseSequenceOp>(typeConverter, ctx), _log(log) {
        setDebugName("ReverseSequenceRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::ReverseSequenceOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReverseSequenceRewrite::matchAndRewrite(VPU::ReverseSequenceOp origOp, OpAdaptor newArgs,
                                                            mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found ReverseSequence Operation '{0}'", origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    auto origSeqLengthShapeType = origOp.seq_length().getType().cast<mlir::ShapedType>();
    auto newSeqLengthShapeType =
            origSeqLengthShapeType.clone(origSeqLengthShapeType.getShape(), mlir::Float16Type::get(getContext()));
    auto memRefType = typeConverter->convertType(newSeqLengthShapeType);
    auto allocOp = rewriter.create<mlir::memref::AllocOp>(origOp->getLoc(), memRefType.cast<mlir::MemRefType>());

    auto convertOp = rewriter.create<VPUIP::ConvertUPAOp>(origOp->getLoc(), newArgs.seq_length(), allocOp.memref());

    auto resultBufs = allocateBuffers(_log, origOp->getLoc(), rewriter, *typeConverter, origOp->getOpResults());

    rewriter.replaceOpWithNewOp<VPUIP::ReverseSequenceUPAOp>(origOp, newArgs.data(), convertOp.output(), resultBufs[0],
                                                             origOp.seq_axisAttr(), origOp.batch_axisAttr());

    return mlir::success();
}

//
// LSTMCellRewrite
//

class LSTMCellRewrite final : public mlir::OpConversionPattern<VPU::LSTMCellOp> {
public:
    LSTMCellRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::LSTMCellOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::LSTMCellOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LSTMCellRewrite::matchAndRewrite(VPU::LSTMCellOp origOp, OpAdaptor newArgs,
                                                     mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found LSTMCell Operation '{0}'", origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    // Concatenate 'weights' and 'recurrenceWeights' into single buffer

    const auto srcWeights = typeConverter->materializeSourceConversion(rewriter, origOp->getLoc(),
                                                                       origOp.weights().getType(), newArgs.weights());
    const auto srcRecurrenceWeights = typeConverter->materializeSourceConversion(
            rewriter, origOp->getLoc(), origOp.recurrenceWeights().getType(), newArgs.recurrenceWeights());

    auto srcConcatenatedWeights =
            rewriter.create<VPU::ConcatOp>(origOp->getLoc(), mlir::ValueRange{srcWeights, srcRecurrenceWeights}, 1);

    const auto targetConcatenatedWeights = typeConverter->materializeTargetConversion(
            rewriter, origOp->getLoc(), typeConverter->convertType(srcConcatenatedWeights.getType()),
            srcConcatenatedWeights.output());

    auto resultBufs = allocateBuffers(_log, origOp->getLoc(), rewriter, *typeConverter, origOp->getOpResults());

    rewriter.replaceOpWithNewOp<VPUIP::LSTMCellUPAOp>(origOp, newArgs.inputData(), newArgs.initialHiddenState(),
                                                      newArgs.initialCellState(), targetConcatenatedWeights,
                                                      newArgs.biases(), resultBufs[0], resultBufs[1]);

    return mlir::success();
}

//
// LSTMSequenceRewrite
//

class LSTMSequenceRewrite final : public mlir::OpConversionPattern<VPU::LSTMSequenceOp> {
public:
    LSTMSequenceRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::LSTMSequenceOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::LSTMSequenceOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LSTMSequenceRewrite::matchAndRewrite(VPU::LSTMSequenceOp origOp, OpAdaptor newArgs,
                                                         mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found LSTMSequence Operation '{0}'", origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    // Concatenate 'weights' and 'recurrenceWeights' into single buffer

    const auto srcWeights = typeConverter->materializeSourceConversion(rewriter, origOp->getLoc(),
                                                                       origOp.weights().getType(), newArgs.weights());
    const auto srcRecurrenceWeights = typeConverter->materializeSourceConversion(
            rewriter, origOp->getLoc(), origOp.reccurenceWeights().getType(), newArgs.reccurenceWeights());

    auto srcConcatenatedWeights =
            rewriter.create<VPU::ConcatOp>(origOp->getLoc(), mlir::ValueRange{srcWeights, srcRecurrenceWeights}, 2);

    const auto targetConcatenatedWeights = typeConverter->materializeTargetConversion(
            rewriter, origOp->getLoc(), typeConverter->convertType(srcConcatenatedWeights.getType()),
            srcConcatenatedWeights.output());

    auto resultBufs = allocateBuffers(_log, origOp->getLoc(), rewriter, *typeConverter, origOp->getOpResults());

    rewriter.replaceOpWithNewOp<VPUIP::LSTMSequenceUPAOp>(origOp, newArgs.inputData(), newArgs.initialHiddenState(),
                                                          newArgs.initialCellState(), targetConcatenatedWeights,
                                                          newArgs.biases(), resultBufs[0], resultBufs[1], resultBufs[2],
                                                          origOp.sequenceLengthAttr(), origOp.directionAttr());

    return mlir::success();
}

//
// FakeQuantizeRewrite
//

class FakeQuantizeRewrite final : public mlir::OpConversionPattern<VPU::FakeQuantizeOp> {
public:
    FakeQuantizeRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::FakeQuantizeOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::FakeQuantizeOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FakeQuantizeRewrite::matchAndRewrite(VPU::FakeQuantizeOp origOp, OpAdaptor newArgs,
                                                         mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found FakeQuantize Operation '{0}'", origOp->getLoc());

    auto inLowConst = newArgs.input_low().getDefiningOp<Const::DeclareOp>();
    auto inHighConst = newArgs.input_high().getDefiningOp<Const::DeclareOp>();
    auto outLowConst = newArgs.output_low().getDefiningOp<Const::DeclareOp>();
    auto outHighConst = newArgs.output_high().getDefiningOp<Const::DeclareOp>();

    if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
        return matchFailed(rewriter, origOp, "Got non constant parameters");
    }

    auto outputBuffers = allocateBuffers(_log, origOp->getLoc(), rewriter, *typeConverter, {origOp.output()});

    rewriter.replaceOpWithNewOp<VPUIP::FakeQuantizeUPAOp>(
            origOp, newArgs.input(), outputBuffers[0], origOp.levelsAttr(), inLowConst.contentAttr(),
            inHighConst.contentAttr(), outLowConst.contentAttr(), outHighConst.contentAttr());

    return mlir::success();
}

//
// FullyConnectedRewrite
//

class FullyConnectedRewrite final : public mlir::OpConversionPattern<VPU::FullyConnectedOp> {
public:
    FullyConnectedRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::FullyConnectedOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::FullyConnectedOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FullyConnectedRewrite::matchAndRewrite(VPU::FullyConnectedOp origOp, OpAdaptor newArgs,
                                                           mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found FullyConnected Operation '{0}'", origOp->getLoc());

    auto outputBuffers = allocateBuffers(_log, origOp->getLoc(), rewriter, *typeConverter, {origOp.output()});

    if (origOp.bias() == nullptr) {
        rewriter.replaceOpWithNewOp<VPUIP::FullyConnectedUPAOp>(origOp, newArgs.input(), newArgs.weights(), nullptr,
                                                                outputBuffers[0]);
        return mlir::success();
    }

    const auto origBiasType = newArgs.bias().getType().cast<vpux::NDTypeInterface>();

    const auto origBiasShape = origBiasType.getShape().raw();
    VPUX_THROW_UNLESS(origBiasShape[0] == 1, "Biases batch size is not equal 1");

    const std::array<int64_t, 1> newBiasShape = {origBiasShape[1]};
    const auto newBiasType = origBiasType.changeShape(ShapeRef(newBiasShape));

    auto newBias = rewriter.create<VPUIP::GenericReshapeOp>(origOp->getLoc(), newBiasType, newArgs.bias());

    rewriter.replaceOpWithNewOp<VPUIP::FullyConnectedUPAOp>(origOp, newArgs.input(), newArgs.weights(),
                                                            newBias.output(), outputBuffers[0]);

    return mlir::success();
}

//
// RewriteConvolution
//

class RewriteConvolution final : public mlir::OpConversionPattern<VPU::ConvolutionOp> {
public:
    RewriteConvolution(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::ConvolutionOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::ConvolutionOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult RewriteConvolution::matchAndRewrite(VPU::ConvolutionOp origOp, OpAdaptor newArgs,
                                                        mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found Convolution Operation '{0}'", origOp->getLoc());

    auto outputBuffers = allocateBuffers(_log, origOp->getLoc(), rewriter, *typeConverter, {origOp.output()});

    const int64_t groups = 1;
    if (origOp.bias() == nullptr) {
        rewriter.replaceOpWithNewOp<VPUIP::ConvolutionUPAOp>(origOp, newArgs.input(), newArgs.filter(), nullptr,
                                                             outputBuffers[0], origOp.strides(), origOp.dilations(),
                                                             origOp.pads_begin(), origOp.pads_end(), groups);
        return mlir::success();
    }

    const auto origBiasType = newArgs.bias().getType().cast<vpux::NDTypeInterface>();
    const auto origBiasShape = origBiasType.getShape().raw();

    const std::array<int64_t, 1> newBiasShape = {origBiasShape[1]};
    const auto newBiasType = origBiasType.changeShape(ShapeRef(newBiasShape));
    auto newBias = rewriter.create<VPUIP::GenericReshapeOp>(origOp->getLoc(), newBiasType, newArgs.bias());

    rewriter.replaceOpWithNewOp<VPUIP::ConvolutionUPAOp>(origOp, newArgs.input(), newArgs.filter(), newBias.output(),
                                                         outputBuffers[0], origOp.strides(), origOp.dilations(),
                                                         origOp.pads_begin(), origOp.pads_end(), groups);
    return mlir::success();
}

//
// TopKRewrite
//

class TopKRewrite final : public mlir::OpConversionPattern<VPU::TopKOp> {
public:
    TopKRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::TopKOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::TopKOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult TopKRewrite::matchAndRewrite(VPU::TopKOp origOp, OpAdaptor newArgs,
                                                 mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found TopK Operation '{0}'", origOp->getLoc());

    auto outputValuesBuffers =
            allocateBuffers(_log, origOp->getLoc(), rewriter, *typeConverter, {origOp.output_values()});
    auto targetShapesBuffers =
            allocateBuffers(_log, origOp->getLoc(), rewriter, *typeConverter, {origOp.target_shape()});

    rewriter.replaceOpWithNewOp<VPUIP::TopKUPAOp>(origOp, newArgs.input(), newArgs.k(), outputValuesBuffers[0],
                                                  targetShapesBuffers[0], origOp.axis(), origOp.mode(), origOp.sort(),
                                                  origOp.element_type());

    return mlir::success();
}

//
// StubRewrite
//

class StubRewrite final : public mlir::OpConversionPattern<VPU::StubOp> {
public:
    StubRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::StubOp>(typeConverter, ctx), _log(log) {
        this->setDebugName("StubRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::StubOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult StubRewrite::matchAndRewrite(VPU::StubOp origOp, OpAdaptor newArgs,
                                                 mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found Stub Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    SmallVector<mlir::Type> outputTypes;
    for (auto out : origOp.getResults()) {
        outputTypes.push_back(typeConverter->convertType(out.getType()));
    }

    rewriter.replaceOpWithNewOp<VPUIP::StubOp>(origOp, outputTypes, newArgs.getOperands());

    return mlir::success();
}

//
// GroupSparseTensorRewriter
//

class GroupSparseTensorRewriter final : public mlir::OpConversionPattern<VPU::GroupSparseTensorOp> {
    using OpAdaptor = typename mlir::OpConversionPattern<VPU::GroupSparseTensorOp>::OpAdaptor;

public:
    GroupSparseTensorRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::GroupSparseTensorOp>(typeConverter, ctx), _log(log) {
        this->setDebugName("GroupSparseTensorRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::GroupSparseTensorOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult GroupSparseTensorRewriter::matchAndRewrite(VPU::GroupSparseTensorOp origOp, OpAdaptor newArgs,
                                                               mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found GroupSparseTensorOp Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    VPUIP::CompressionSchemeAttr compressionScheme = nullptr;
    if (origOp.compression_schemeAttr() != nullptr) {
        auto origCompression = origOp.compression_schemeAttr();
        compressionScheme =
                VPUIP::CompressionSchemeAttr::get(origCompression.getContext(), origCompression.getAxis(),
                                                  origCompression.getNumElems(), origCompression.getAlignment());
    }

    rewriter.replaceOpWithNewOp<VPUIP::GroupSparseBufferOp>(origOp, newArgs.data(), newArgs.sparsityMap(),
                                                            newArgs.storageElementTable(), origOp.is_weightsAttr(),
                                                            compressionScheme);

    return mlir::success();
}

// RewriteNonMaxSuppression
//

class NonMaxSuppressionRewrite final : public mlir::OpConversionPattern<VPU::NonMaxSuppressionOp> {
public:
    NonMaxSuppressionRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::NonMaxSuppressionOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NonMaxSuppressionOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult NonMaxSuppressionRewrite::matchAndRewrite(VPU::NonMaxSuppressionOp origOp, OpAdaptor newArgs,
                                                              mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found NonMaxSuppression Operation '{0}'", origOp->getLoc());

    auto resultBufs = allocateBuffers(_log, origOp->getLoc(), rewriter, *typeConverter, origOp->getOpResults());

    rewriter.replaceOpWithNewOp<VPUIP::NonMaxSuppressionUPAOp>(
            origOp, newArgs.in_box_coords(), newArgs.in_box_scores(), resultBufs[0], resultBufs[1], resultBufs[2],
            origOp.box_encodingAttr(), origOp.sort_result_descendingAttr(),
            origOp.max_output_boxes_per_class_valueAttr(), origOp.iou_threshold_valueAttr(),
            origOp.score_threshold_valueAttr(), origOp.soft_nms_sigma_valueAttr());

    _log.trace("Replaced with 'VPUIP.NonMaxSuppressionOp'");

    return mlir::success();
}

//
// StorageElementTableRewriter
//

class StorageElementTableRewriter final : public mlir::OpConversionPattern<VPU::StorageElementTableOp> {
    using OpAdaptor = typename mlir::OpConversionPattern<VPU::StorageElementTableOp>::OpAdaptor;

public:
    StorageElementTableRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::StorageElementTableOp>(typeConverter, ctx), _log(log) {
        this->setDebugName("StorageElementTableRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::StorageElementTableOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult StorageElementTableRewriter::matchAndRewrite(VPU::StorageElementTableOp origOp,
                                                                 OpAdaptor /*newArgs*/,
                                                                 mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found StorageElementTableOp Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPUIP::StorageElementTableOp>(origOp, origOp.seDepthAttr(), origOp.seSizeAttr(),
                                                              origOp.heightAttr(), origOp.widthAttr(),
                                                              origOp.base_ptrsAttr());

    return mlir::success();
}

//
// ShapeCastRewrite
//

class ShapeCastRewrite final : public mlir::OpConversionPattern<VPU::ShapeCastOp> {
public:
    ShapeCastRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::ShapeCastOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::ShapeCastOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ShapeCastRewrite::matchAndRewrite(VPU::ShapeCastOp origOp, OpAdaptor newArgs,
                                                      mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found ShapeCast Operation '{0}'", origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPUIP::ShapeCastOp>(origOp, newArgs.source(), newArgs.shape());

    _log.trace("Replaced with 'VPUIP.ShapeCastOp'");

    return mlir::success();
}

//
// LayoutCastRewrite
//

class LayoutCastRewrite final : public mlir::OpConversionPattern<VPU::LayoutCastOp> {
public:
    LayoutCastRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::LayoutCastOp>(typeConverter, ctx), _log(log) {
        setDebugName("LayoutCastRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::LayoutCastOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LayoutCastRewrite::matchAndRewrite(VPU::LayoutCastOp origOp, OpAdaptor newArgs,
                                                       mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found LayoutCast Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    const auto newOutType = typeConverter->convertType(origOp.getType());

    const auto outOrder = DimsOrder::fromValue(origOp.output());
    const auto outMap = outOrder.toAffineMap(origOp.getContext());
    const auto mapAttr = mlir::AffineMapAttr::get(outMap);
    rewriter.replaceOpWithNewOp<VPUIP::PermuteCastOp>(origOp, newOutType, newArgs.input(), origOp.dst_orderAttr(),
                                                      mapAttr);
    return mlir::success();
}

//
// WorkloadCastRewrite
//

class WorkloadCastRewrite final : public mlir::OpConversionPattern<VPU::WorkloadCastOp> {
public:
    WorkloadCastRewrite(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPU::WorkloadCastOp>(typeConverter, ctx), _log(log) {
        setDebugName("WorkloadCastRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::WorkloadCastOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult WorkloadCastRewrite::matchAndRewrite(VPU::WorkloadCastOp origOp, OpAdaptor newArgs,
                                                         mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Found WorkloadCast Operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto* typeConverter = getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter is not set");

    const auto newOutType = typeConverter->convertType(origOp.getType());
    rewriter.replaceOpWithNewOp<VPUIP::WorkloadCastOp>(origOp, newOutType, newArgs.input());

    return mlir::success();
}

//
// ConvertLayers2VPUIPPass
//

class ConvertLayers2VPUIPPass final : public ConvertLayers2VPUIPBase<ConvertLayers2VPUIPPass> {
public:
    explicit ConvertLayers2VPUIPPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertLayers2VPUIPPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    vpux::BufferizeTypeConverter typeConverter;

    const auto isLegalOp = [&](mlir::Operation* op) {
        return typeConverter.isLegal(op);
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalDialect<Const::ConstDialect>(isLegalOp);
    target.addIllegalDialect<VPU::VPUDialect>();
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalDialect<VPURT::VPURTDialect>();
    target.addLegalOp<mlir::FuncOp, mlir::ReturnOp>();
    target.addLegalOp<mlir::memref::AllocOp>();
    target.addLegalOp<VPU::NCEConvolutionOp, VPU::NCEDepthConvolutionOp, VPU::NCEMaxPoolOp, VPU::NCEAveragePoolOp,
                      VPU::NCEEltwiseOp, VPU::NCEPermuteQuantizeOp>();
    target.addLegalOp<VPU::DPUWorkloadOp>();
    target.addLegalOp<VPU::NCEClusterTilingOp, VPU::YieldOp>();
    target.addLegalOp<VPUIP::SwKernelOp>();
    target.markOpRecursivelyLegal<VPUIP::SwKernelOp>([&](mlir::Operation*) {
        return true;
    });
    vpux::populateBufferizeMaterializationLegality(target);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<LayerRewrite>(typeConverter, &ctx, _log);
    patterns.add<ReshapeRewrite<VPU::AffineReshapeOp>>(typeConverter, &ctx, _log);
    patterns.add<ReshapeRewrite<VPU::ReshapeOp>>(typeConverter, &ctx, _log);
    patterns.add<ReshapeRewrite<VPU::SqueezeOp>>(typeConverter, &ctx, _log);
    patterns.add<ReshapeRewrite<VPU::UnsqueezeOp>>(typeConverter, &ctx, _log);
    patterns.add<SliceRewrite>(typeConverter, &ctx, _log);
    patterns.add<SplitRewrite>(typeConverter, &ctx, _log);
    patterns.add<ConcatRewrite>(typeConverter, &ctx, _log);
    patterns.add<PermuteCastRewrite>(typeConverter, &ctx, _log);
    patterns.add<QuantizeCastRewriter>(typeConverter, &ctx, _log);
    patterns.add<DistributedCastRewriter>(typeConverter, &ctx, _log);
    patterns.add<ReverseSequenceRewrite>(typeConverter, &ctx, _log);
    patterns.add<LSTMCellRewrite>(typeConverter, &ctx, _log);
    patterns.add<LSTMSequenceRewrite>(typeConverter, &ctx, _log);
    patterns.add<FakeQuantizeRewrite>(typeConverter, &ctx, _log);
    patterns.add<FullyConnectedRewrite>(typeConverter, &ctx, _log);
    patterns.add<RewriteConvolution>(typeConverter, &ctx, _log);
    patterns.add<TopKRewrite>(typeConverter, &ctx, _log);
    patterns.add<StubRewrite>(typeConverter, &ctx, _log);
    patterns.add<GroupSparseTensorRewriter>(typeConverter, &ctx, _log);
    patterns.add<NonMaxSuppressionRewrite>(typeConverter, &ctx, _log);
    patterns.add<StorageElementTableRewriter>(typeConverter, &ctx, _log);
    patterns.add<ShapeCastRewrite>(typeConverter, &ctx, _log);
    patterns.add<LayoutCastRewrite>(typeConverter, &ctx, _log);
    patterns.add<WorkloadCastRewrite>(typeConverter, &ctx, _log);
    Const::ConstDialect::populateBufferizePatterns(patterns, typeConverter, _log);

    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertLayers2VPUIPPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertLayers2VPUIPPass(Logger log) {
    return std::make_unique<ConvertLayers2VPUIPPass>(log);
}
