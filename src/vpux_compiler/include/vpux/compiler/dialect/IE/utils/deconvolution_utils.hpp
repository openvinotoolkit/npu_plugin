//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

namespace vpux {
namespace IE {

mlir::FailureOr<Const::DeclareOp> getConstFilter(IE::DeconvolutionOp deconv);
mlir::LogicalResult canConvertDeconvToConv(IE::DeconvolutionOp deconv);
mlir::LogicalResult canConvertGroupDeconvToGroupConv(IE::GroupDeconvolutionOp groupDeconv);

// For 2D GroupDeconvolution:
// input tensor layout is [N, C_IN * GROUPS, H, W]
// kernel tensor layout is [GROUPS, C_IN, C_OUT, kH, kW]
const int64_t GROUP_DECONV_C_IN_DIM_INDEX = 1;
const int64_t GROUP_DECONV_C_OUT_DIM_INDEX = 2;
const int64_t GROUP_DECONV_KY_DIM_INDEX = 3;
const int64_t GROUP_DECONV_KX_DIM_INDEX = 4;

template <class ConcreteOp>
mlir::FailureOr<mlir::Value> createUpsampling(mlir::PatternRewriter& rewriter, ConcreteOp origOp,
                                              vpux::Shape& padsOutput, bool isGroupDeconv) {
    auto ctx = rewriter.getContext();

    const auto padsBeginVector = Shape(parseIntArrayAttr<int64_t>(origOp.pads_begin()));
    const auto padsEndVector = Shape(parseIntArrayAttr<int64_t>(origOp.pads_end()));
    const auto stridesVector = Shape(parseIntArrayAttr<int64_t>(origOp.strides()));

    int64_t padL, padR, padT, padB;

    if (isGroupDeconv) {
        auto origFilterShape =
                to_small_vector(origOp.filter().getType().template dyn_cast<mlir::ShapedType>().getShape());
        if (origFilterShape.size() != 5) {
            return errorAt(origOp, "Only 2D GroupDeconvolution is supported, expected 5D filter but got {0}",
                           origFilterShape.size());
        }

        padL = origFilterShape[GROUP_DECONV_KX_DIM_INDEX] - 1 - padsBeginVector[Dims4D::PadsBegin::Left];
        padR = origFilterShape[GROUP_DECONV_KX_DIM_INDEX] - 1 - padsEndVector[Dims4D::PadsEnd::Right];
        padT = origFilterShape[GROUP_DECONV_KY_DIM_INDEX] - 1 - padsBeginVector[Dims4D::PadsBegin::Top];
        padB = origFilterShape[GROUP_DECONV_KY_DIM_INDEX] - 1 - padsEndVector[Dims4D::PadsEnd::Bottom];
    } else {
        auto filterShape = getShape(origOp.filter()).toValues();
        if (filterShape.size() != 4) {
            return errorAt(origOp, "Only 2D Deconvolution is supported, expected 4D filter but got {0}",
                           filterShape.size());
        }

        padL = filterShape[Dims4D::Filter::KX] - 1 - padsBeginVector[Dims4D::PadsBegin::Left];
        padR = filterShape[Dims4D::Filter::KX] - 1 - padsEndVector[Dims4D::PadsEnd::Right];
        padT = filterShape[Dims4D::Filter::KY] - 1 - padsBeginVector[Dims4D::PadsBegin::Top];
        padB = filterShape[Dims4D::Filter::KY] - 1 - padsEndVector[Dims4D::PadsEnd::Bottom];
    }

    // Output padding refers to copying convolutional input data. If the value of output padding is less than the value
    // of PadR&PadB, the copied data will be 0, so it can be merged with PadR&PadB.
    if ((padsOutput[Dims4D::PadsOutput::Y] > 0) && (padsOutput[Dims4D::PadsOutput::Y] <= padB)) {
        padB += padsOutput[Dims4D::PadsOutput::Y];
        padsOutput[Dims4D::PadsOutput::Y] = 0;
    }
    if ((padsOutput[Dims4D::PadsOutput::X] > 0) && (padsOutput[Dims4D::PadsOutput::X] <= padR)) {
        padR += padsOutput[Dims4D::PadsOutput::X];
        padsOutput[Dims4D::PadsOutput::X] = 0;
    }

    auto padChannelAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    auto padHeightAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{padT, padB});
    auto padWidthAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{padL, padR});
    auto padAttr = IE::UpsamplingPadAttr::get(ctx, padChannelAttr, padHeightAttr, padWidthAttr);

    if ((padL < 0) || (padR < 0) || (padT < 0) || (padB < 0)) {
        return errorAt(origOp, "Upsampling layer does not support negative paddings");
    }

    auto upsamplingFactor = getIntArrayAttr(
            ctx, SmallVector<int64_t>{stridesVector[Dims4D::Strides::X], stridesVector[Dims4D::Strides::Y], 1});

    return rewriter.create<IE::UpsamplingOp>(origOp->getLoc(), origOp.feature(), upsamplingFactor, padAttr).output();
}

mlir::Value createPadding(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value input, Dim axis,
                          int64_t nums, IE::FakeQuantizeOp inputFQ);

}  // namespace IE
}  // namespace vpux
