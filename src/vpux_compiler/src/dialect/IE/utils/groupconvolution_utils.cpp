//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/groupconvolution_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
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

    const auto group = groupconv.groups().value();
    const auto filterShape = getShape(groupconv.filter());
    if (filterShape[Dims4D::Filter::OC] == group) {
        logCb(formatv("Conversion is not needed for dw conv"));
        return mlir::failure();
    }

    // Channel alignment is not checked here because experiments show that NCE is still able to provide better
    // performance than SHAVE even if channel expand is done.

    const auto arch = vpux::VPU::getArch(groupconv.getOperation());
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto kernelStrides = parseIntArrayAttr<int64_t>(groupconv.strides());
    const auto kernelStridesShape = Shape(kernelStrides);
    const auto SY = kernelStridesShape[Dims4D::Strides::Y];
    const auto SX = kernelStridesShape[Dims4D::Strides::X];
    const auto pads = PadInfo(groupconv.pads_begin(), groupconv.pads_end());
    if (!VPU::NCEInvariant::isAttrsSupported(arch, KY, KX, SY, SX, pads.top, pads.bottom, pads.left, pads.right,
                                             logCb)) {
        return mlir::failure();
    }

    return mlir::success();
}

bool groupConvIsEltwise(IE::GroupConvolutionOp convOp) {
    // check kernel size is 1x1
    auto filterShape = getShape(convOp.filter());
    if (filterShape[Dims4D::Filter::KX] != 1 || filterShape[Dims4D::Filter::KX] != 1 ||
        filterShape[Dims4D::Filter::OC] != convOp.groups().getValue()) {
        return false;
    }
    // if there is stride > 1, it can not consider to be an eltwise op
    const auto greaterThanOne = [](auto stride) {
        return stride > 1;
    };
    auto stridesGreaterThanOne = llvm::any_of(parseIntArrayAttr<int64_t>(convOp.strides()), greaterThanOne);
    if (stridesGreaterThanOne) {
        return false;
    }
    // check input const is single data or not
    mlir::SmallVector<Const::DeclareOp> constInputOps;
    constInputOps.push_back(convOp.filter().getDefiningOp<Const::DeclareOp>());
    if (convOp.bias()) {
        constInputOps.push_back(convOp.bias().getDefiningOp<Const::DeclareOp>());
    }
    return llvm::all_of(constInputOps, [](Const::DeclareOp constOp) {
        auto realDataSizeResult = getBaseContentNumElements(constOp);
        return mlir::succeeded(realDataSizeResult) && realDataSizeResult.value() == 1;
    });
}
}  // namespace IE
}  // namespace vpux
