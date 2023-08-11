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
                                   LogCb logCb) {
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
    const auto inputChannelAligment =
            !isChannelMajor ? VPU::NCEConvolutionOp::getInputChannelAlignmentImpl(inputType) : 1;

    if (checkChannelAlignment) {
        if (!NCEInvariant::isInputActTypeSupported(arch, inputType, inputChannelAligment, true) ||
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
