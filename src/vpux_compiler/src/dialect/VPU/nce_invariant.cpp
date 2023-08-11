//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"

#include "vpux/utils/core/numeric.hpp"

using namespace vpux;

//
// Precision checks
//

bool vpux::VPU::NCEInvariant::isPrecisionSupported(ArchKind arch, mlir::ValueRange vals, LogCb logCb) {
    for (const auto& val : vals) {
        const auto elemType = val.getType().cast<vpux::NDTypeInterface>().getElementType();

        if (elemType.isBF16() && arch != ArchKind::VPUX37XX) {
            logCb(formatv("BF16 is only supported by VPUX37XX"));
            return false;
        }
    }

    return true;
}

//
// Fuse PadOp check
//

bool vpux::VPU::NCEInvariant::verifyPads(int64_t KY, int64_t KX, int64_t padTop, int64_t padBottom, int64_t padLeft,
                                         int64_t padRight, LogCb logCb) {
    if (padTop < 0 || padTop > KY / 2) {
        logCb(formatv("Unsupported padding '{0}', must be in range [0, {1}]", padTop, KY / 2));
        return false;
    }
    if (padBottom < 0 || padBottom > KY / 2) {
        logCb(formatv("Unsupported padding '{0}', must be in range [0, {1}]", padBottom, KY / 2));
        return false;
    }
    if (padLeft < 0 || padLeft > KX / 2) {
        logCb(formatv("Unsupported padding '{0}', must be in range [0, {1}]", padLeft, KX / 2));
        return false;
    }
    if (padRight < 0 || padRight > KX / 2) {
        logCb(formatv("Unsupported padding '{0}', must be in range [0, {1}]", padRight, KX / 2));
        return false;
    }

    return true;
}

bool vpux::VPU::NCEInvariant::verifyPads(mlir::ArrayAttr kernelSizeAttr, mlir::ArrayAttr padBeginAttr,
                                         mlir::ArrayAttr padEndAttr, LogCb logCb) {
    const auto kernelSize = parseIntArrayAttr<int64_t>(kernelSizeAttr);
    const auto KY = kernelSize[kernelSize.size() == 4 ? (Dims4D::Filter::KY.ind()) : (Dims4D::Kernel::Y.ind())];
    const auto KX = kernelSize[kernelSize.size() == 4 ? (Dims4D::Filter::KX.ind()) : (Dims4D::Kernel::X.ind())];

    const auto padsBegin = parseIntArrayAttr<int64_t>(padBeginAttr);
    const auto padsEnd = parseIntArrayAttr<int64_t>(padEndAttr);
    const auto padTop = padsBegin[Dims4D::PadsBegin::Top.ind()];
    const auto padLeft = padsBegin[Dims4D::PadsBegin::Left.ind()];
    const auto padBottom = padsEnd[Dims4D::PadsEnd::Bottom.ind()];
    const auto padRight = padsEnd[Dims4D::PadsEnd::Right.ind()];

    return verifyPads(KY, KX, padTop, padBottom, padLeft, padRight, logCb);
}

//
// Attributes checks
//

bool vpux::VPU::NCEInvariant::isAttrsSupported(ArchKind arch, int64_t KY, int64_t KX, int64_t SY, int64_t SX,
                                               int64_t padTop, int64_t padBottom, int64_t padLeft, int64_t padRight,
                                               LogCb logCb) {
    static const int64_t NCE_MAX_STRIDE_SIZE = 8;

    if (KY > MAX_KERNEL_SIZE || KY <= 0) {
        logCb(formatv("Unsupported kernel height dimension '{0}', must be in range [1, {1}]", KY, MAX_KERNEL_SIZE));
        return false;
    }
    if (KX > MAX_KERNEL_SIZE || KX <= 0) {
        logCb(formatv("Unsupported kernel width dimension '{0}', must be in range [1, {1}]", KX, MAX_KERNEL_SIZE));
        return false;
    }

    if (SX != SY && arch != VPU::ArchKind::VPUX37XX) {
        logCb(formatv("Asymmetric strides are not supported"));
        return false;
    }
    if (SY > NCE_MAX_STRIDE_SIZE || SY <= 0) {
        logCb(formatv("Unsupported stride height dimension '{0}', must be in range [1, {1}]", SY, NCE_MAX_STRIDE_SIZE));
        return false;
    }
    if (SX > NCE_MAX_STRIDE_SIZE || SX <= 0) {
        logCb(formatv("Unsupported stride width dimension '{0}', must be in range [1, {1}]", SX, NCE_MAX_STRIDE_SIZE));
        return false;
    }

    return verifyPads(KY, KX, padTop, padBottom, padLeft, padRight, logCb);
}

//
// Activation type checks
//

bool vpux::VPU::NCEInvariant::isAligned(vpux::NDTypeInterface type, int64_t alignment, ArchKind arch, LogCb logCb) {
    const auto shape = type.getShape();
    const auto order = type.getDimsOrder();
    const auto memShape = order.toMemoryOrder(shape);

    const bool supportsSuperDense = VPU::NCEInvariant::isSuperdenseSupported(arch);
    // In super-dense mode only channels must be aligned.
    const auto channels = shape[Dims4D::Act::C];
    if (supportsSuperDense && channels % alignment == 0) {
        return true;
    }

    const auto innerDim = memShape.back();
    if (innerDim % alignment != 0) {
        logCb(formatv("Activation inner dimension '{0}' is not aligned to '{1}'", innerDim, alignment));
        return false;
    }

    return true;
}

int64_t vpux::VPU::NCEInvariant::getAlignment(mlir::Type elemType) {
    const auto typeSizeInBits = static_cast<Bit>(getElemTypeSize(elemType));
    return std::max<int64_t>(128 / typeSizeInBits.count(), 16);
}

bool vpux::VPU::NCEInvariant::isOutputActTypeSupported(vpux::NDTypeInterface type, int64_t alignment, LogCb logCb) {
    if (type.getRank() != 4) {
        logCb(formatv("Ouput activation has unsupported rank: {0}", type.getRank()));
        return false;
    }

    const auto OC = type.getShape()[Dims4D::Act::C];
    if (OC % alignment != 0) {
        logCb(formatv("Output input channels '{0}' are not aligned to '{1}'", OC, alignment));
        return false;
    }

    return true;
}

bool vpux::VPU::NCEInvariant::isInputActTypeSupported(ArchKind arch, vpux::NDTypeInterface type, int64_t alignment,
                                                      bool supportsInputActCompression, LogCb logCb) {
    if (type.getRank() != 4) {
        logCb(formatv("Input activation has unsupported rank: {0}", type.getRank()));
        return false;
    }

    if ((arch == ArchKind::VPUX37XX) && supportsInputActCompression) {
        const auto IC = type.getShape()[Dims4D::Act::C];
        return IC == VPU_COMPRESSED_INPUT_CHANNEL_NUM || isAligned(type, alignment, arch, logCb);
    }

    return isAligned(type, alignment, arch, logCb);
}

//
// WeightsTable information
//

Byte vpux::VPU::NCEInvariant::getWeightsTableSize(int64_t OC) {
    return OC * WEIGHT_TABLE_NUM_ELEMENTS_PER_OC * 4_Byte;
}

//
// Channel major Convolution
//

bool vpux::VPU::NCEInvariant::isChannelMajorCompatible(ArchKind arch, vpux::NDTypeInterface inputType) {
    if (arch != ArchKind::VPUX30XX && arch != ArchKind::VPUX311X) {
        return false;
    }

    const auto inputShape = inputType.getShape();
    if (inputShape.size() < 4) {
        return false;
    }

    const auto IC = inputShape[Dims4D::Act::C];
    const auto IW = inputShape[Dims4D::Act::W];

    return (IC < NCEInvariant::KMB_CMCONV_CHANNELS_LIMIT) && (IW % NCEInvariant::KMB_CMCONV_WIDTH_ALIGNMENT == 0);
}

//
// Common utility for AvgPool, MaxPool, Eltwise and DWConv
//

bool vpux::VPU::NCEInvariant::checkLayouts(mlir::TypeRange operandTypes, mlir::TypeRange resultTypes,
                                           const VPU::ArchKind& arch, const unsigned numInputOperands, LogCb logCb) {
    for (unsigned opIdx = 0; opIdx < numInputOperands; opIdx++) {
        const auto actualInLayout = operandTypes[opIdx].cast<vpux::NDTypeInterface>().getDimsOrder();
        const auto& expectedInLayout = DimsOrder::NHWC;
        if (actualInLayout != expectedInLayout) {
            logCb(formatv("Unsupported input layout. Expected: {0}, got: {1}", expectedInLayout, actualInLayout));
            return false;
        }
    }

    for (auto resultType : resultTypes) {
        const auto actualOutLayout = resultType.cast<vpux::NDTypeInterface>().getDimsOrder();
        const auto& expectedOutLayout = DimsOrder::NHWC;
        if (arch != VPU::ArchKind::VPUX37XX &&
            actualOutLayout != expectedOutLayout) {
            logCb(formatv("Unsupported output layout. Expected: {0}, got: {1}", expectedOutLayout, actualOutLayout));
            return false;
        }
    }

    return true;
}

//
// Compress Convolution
//

bool vpux::VPU::NCEInvariant::isCompressConvolution(ArchKind arch, mlir::Operation* op) {
    if (arch != ArchKind::VPUX37XX) {
        return false;
    }

    if (auto origOp = mlir::dyn_cast<vpux::VPU::NCEConvolutionOp>(op)) {
        if (DimsOrder::fromValue(origOp.input()) != DimsOrder::NHWC) {
            return false;
        }

        const auto inputType = origOp.input().getType().cast<vpux::NDTypeInterface>();
        const auto IC = inputType.getShape()[Dims4D::Act::C];
        if (IC != VPU::NCEInvariant::VPU_COMPRESSED_INPUT_CHANNEL_NUM) {
            return false;
        }

        return true;
    }
    if (auto origOp = mlir::dyn_cast<vpux::VPU::NCECompressConvolutionOp>(op)) {
        return true;
    }

    return false;
}

bool vpux::VPU::NCEInvariant::isSuperdenseSupported(const VPU::ArchKind arch) {
    const llvm::DenseSet<VPU::ArchKind> compatibleTargets = {
            VPU::ArchKind::VPUX37XX,
    };
    return compatibleTargets.contains(arch);
}
