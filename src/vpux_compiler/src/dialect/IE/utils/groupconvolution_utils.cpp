//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/groupconvolution_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/utils/core/logger.hpp"

namespace vpux {
namespace IE {

mlir::LogicalResult canConvertGroupConvToConv(IE::GroupConvolutionOp groupconv) {
    LogCb logCb = globalLogCb;
    if (!groupconv.groups().has_value()) {
        logCb(formatv("Grouped convolution does not have groups attribute"));
        return mlir::failure();
    }

    const auto inputType = groupconv.input().getType().cast<NDTypeInterface>();
    const auto filterType = groupconv.filter().getType().cast<NDTypeInterface>();
    const auto outputType = groupconv.output().getType().cast<NDTypeInterface>();
    if (inputType.getRank() != 4) {
        logCb(formatv("Only 4D tensors are supported"));
        return mlir::failure();
    }
    if (outputType.getRank() != 4) {
        logCb(formatv("Only 4D tensors are supported"));
        return mlir::failure();
    }
    if (filterType.getRank() != 4) {
        logCb(formatv("Only 4D tensors are supported"));
        return mlir::failure();
    }

    const auto dilation = parseIntArrayAttr<int64_t>(groupconv.dilations());
    if (dilation.size() != 2) {
        logCb(formatv("Expected dilations size to be 2, got '{0}'", dilation.size()));
        return mlir::failure();
    }
    if (dilation[0] != 1 || dilation[1] != 1) {
        logCb(formatv("Dilated convolution is not supported"));
        return mlir::failure();
    }

    const auto arch = vpux::VPU::getArch(groupconv.getOperation());
    const auto group = groupconv.groups().value();
    const auto filterShape = getShape(groupconv.filter());
    const auto OC = filterShape[Dims4D::Filter::OC] / group;
    const auto IC = filterShape[Dims4D::Filter::IC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];
    const auto alignment = VPU::NCEInvariant::getAlignment(outputType.getElementType());

    // Currently groupconv is only allowed to be converted if the resulting conv is channel-aligned
    // TODO: More precise condition should be used to determine whether groupconv should run on NCE or on SHAVE
    //       Possible factors: input/ouput size, channel alignment, arch, etc
    if (OC % alignment != 0 || IC % alignment != 0) {
        logCb(formatv("Converted convolutions' channels are not aligned: IC {0}, OC {1}", IC, OC));
        return mlir::failure();
    }

    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(groupconv.strides()));
    const auto SY = kernelStrides[Dims4D::Strides::Y];
    const auto SX = kernelStrides[Dims4D::Strides::X];
    const auto pads = PadInfo(groupconv.pads_begin(), groupconv.pads_end());
    if (!VPU::NCEInvariant::isAttrsSupported(arch, KY, KX, SY, SX, pads.top, pads.bottom, pads.left, pads.right,
                                             logCb)) {
        return mlir::failure();
    }

    return mlir::success();
}
}  // namespace IE
}  // namespace vpux
