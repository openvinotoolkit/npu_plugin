//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"

#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/numeric.hpp"

using namespace vpux;
using namespace VPU;

bool vpux::VPU::isNCEConvSupported(VPU::ArchKind arch, NDTypeInterface inputType, NDTypeInterface filterType,
                                   NDTypeInterface outputType, ArrayRef<int64_t> dilations, int64_t KY, int64_t KX,
                                   int64_t SY, int64_t SX, PadInfo pads, bool checkLayout, bool checkChannelAlignment,
                                   LogCb logCb, bool supportsInputActCompression) {
    if (outputType.getRank() != 4) {
        logCb(formatv("Only 4D tensors are supported"));
        return false;
    }

    if (dilations.size() != 2) {
        logCb(formatv("Expected dilations size to be 2, got '{0}'", dilations.size()));
        return false;
    }
    if (dilations[0] != 1 || dilations[1] != 1) {
        logCb(formatv("Dilated convolution is not supported"));
        return false;
    }

    if (!NCEInvariant::isAttrsSupported(arch, KY, KX, SY, SX, pads.top, pads.bottom, pads.left, pads.right, logCb)) {
        return false;
    }

    const auto inputOrder = inputType.getDimsOrder();
    const auto isChannelMajor = inputOrder == DimsOrder::NCHW;
    const auto inputChannelAlignment =
            !isChannelMajor ? VPU::NCEConvolutionOp::getInputChannelAlignmentImpl(inputType) : 1;

    if (checkChannelAlignment) {
        if (!NCEInvariant::isInputActTypeSupported(arch, inputType, inputChannelAlignment,
                                                   supportsInputActCompression) ||
            !NCEInvariant::isOutputActTypeSupported(outputType,
                                                    VPU::NCEConvolutionOp::getOutputChannelAlignmentImpl(outputType))) {
            logCb(formatv("Misaligned tensor shape"));
            return false;
        }
    }

    if (checkLayout) {
        const auto filterOrder = filterType.getDimsOrder();
        const auto outputOrder = outputType.getDimsOrder();

        if (inputOrder != DimsOrder::NHWC && inputOrder != DimsOrder::NCHW) {
            logCb(formatv("Unsupported input layout '{0}'", inputOrder));
            return false;
        }
        if (filterOrder != DimsOrder::OYXI) {
            logCb(formatv("Unsupported filter layout '{0}'", filterOrder));
            return false;
        }
        if (arch != VPU::ArchKind::VPUX37XX && outputOrder != DimsOrder::NHWC) {
            logCb(formatv("Unsupported output layout '{0}'", outputOrder));
            return false;
        }
    }

    return true;
}

bool vpux::VPU::isSupportedConv(IE::ConvolutionOp op, LogCb logCb, bool checkLayout, bool checkChannelAlignment,
                                bool supportsInputActCompression) {
    const auto arch = getArch(op);

    const auto dilations = parseIntArrayAttr<int64_t>(op.dilations());

    const auto filterShape = getShape(op.filter());
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(op.strides()));
    const auto SY = kernelStrides[Dims4D::Strides::Y];
    const auto SX = kernelStrides[Dims4D::Strides::X];

    const auto pads = PadInfo(op.pads_begin(), op.pads_end());

    const auto inputType = op.input().getType().cast<NDTypeInterface>();
    const auto filterType = op.filter().getType().cast<NDTypeInterface>();
    const auto outputType = op.output().getType().cast<NDTypeInterface>();

    return VPU::isNCEConvSupported(arch, inputType, filterType, outputType, dilations, KY, KX, SY, SX, pads,
                                   checkLayout, checkChannelAlignment, logCb, supportsInputActCompression);
}

mlir::LogicalResult vpux::VPU::verifyConvUtil(mlir::Location loc, VPU::ArchKind arch, Shape filterShape,
                                              Shape kernelStrides, PaddingAttr padAttr, ShapeRef weightsTableShape,
                                              mlir::Value output) {
    const auto logCb = [loc](const formatv_object_base& msg) {
        std::ignore = errorAt(loc, "{0}", msg.str());
    };

    const auto outputShape = getShape(output);
    const auto OC = outputShape[Dims4D::Act::C];

    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto SY = kernelStrides[Dims4D::Strides::Y];
    const auto SX = kernelStrides[Dims4D::Strides::X];

    const auto padTop = padAttr.getTop().getValue().getSExtValue();
    const auto padBottom = padAttr.getBottom().getValue().getSExtValue();
    const auto padLeft = padAttr.getLeft().getValue().getSExtValue();
    const auto padRight = padAttr.getRight().getValue().getSExtValue();

    if (!VPU::NCEInvariant::isAttrsSupported(arch, KY, KX, SY, SX, padTop, padBottom, padLeft, padRight, logCb)) {
        return mlir::failure();
    }

    const auto expectedWeightsTableShape = VPU::NCESparsity::inferWeightsTableShape(OC);

    if (weightsTableShape != expectedWeightsTableShape) {
        return errorAt(loc, "Got wrong shape for 'weightsTable' '{0}', expected '{1}'", weightsTableShape,
                       expectedWeightsTableShape);
    }

    return mlir::success();
}
