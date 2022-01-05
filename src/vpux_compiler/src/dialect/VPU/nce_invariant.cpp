//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"

#include "vpux/utils/core/numeric.hpp"

using namespace vpux;

//
// LogCb
//

void vpux::VPU::NCEInvariant::emptyLogCb(const llvm::formatv_object_base&) {
}

//
// Precision checks
//

bool vpux::VPU::NCEInvariant::isPrecisionSupported(ArchKind arch, mlir::ValueRange vals, LogCb logCb) {
    for (const auto& val : vals) {
        const auto elemType = val.getType().cast<mlir::ShapedType>().getElementType();

        if (elemType.isBF16() && arch != ArchKind::MTL) {
            logCb(llvm::formatv("BF16 is only supported by MTL"));
            return false;
        }
    }

    return true;
}

//
// Attributes checks
//

bool vpux::VPU::NCEInvariant::isAttrsSupported(ArchKind arch, int64_t KY, int64_t KX, int64_t SY, int64_t SX,
                                               int64_t padTop, int64_t padBottom, int64_t padLeft, int64_t padRight,
                                               LogCb logCb) {
    static const int64_t NCE_MAX_KERNEL_SIZE = 11;
    static const int64_t NCE_MAX_STRIDE_SIZE = 8;

    if (KY > NCE_MAX_KERNEL_SIZE || KY <= 0) {
        logCb(llvm::formatv("Unsupported kernel height dimension '{0}', must be in range [1, {1}]", KY,
                            NCE_MAX_KERNEL_SIZE));
        return false;
    }
    if (KX > NCE_MAX_KERNEL_SIZE || KX <= 0) {
        logCb(llvm::formatv("Unsupported kernel width dimension '{0}', must be in range [1, {1}]", KX,
                            NCE_MAX_KERNEL_SIZE));
        return false;
    }

    if (SX != SY && arch != VPU::ArchKind::MTL) {
        logCb(llvm::formatv("Asymmetric strides are not supported"));
        return false;
    }
    if (SY > NCE_MAX_STRIDE_SIZE || SY <= 0) {
        logCb(llvm::formatv("Unsupported stride height dimension '{0}', must be in range [1, {1}]", SY,
                            NCE_MAX_STRIDE_SIZE));
        return false;
    }
    if (SX > NCE_MAX_STRIDE_SIZE || SX <= 0) {
        logCb(llvm::formatv("Unsupported stride width dimension '{0}', must be in range [1, {1}]", SX,
                            NCE_MAX_STRIDE_SIZE));
        return false;
    }

    if (padTop < 0 || (padTop > 1 && padTop > KY / 2)) {
        logCb(llvm::formatv("Unsupported padding '{0}', must be in range [0, {1}]", padTop, KY / 2));
        return false;
    }
    if (padBottom < 0 || (padBottom > 1 && padBottom > KY / 2)) {
        logCb(llvm::formatv("Unsupported padding '{0}', must be in range [0, {1}]", padBottom, KY / 2));
        return false;
    }
    if (padLeft < 0 || (padLeft > 1 && padLeft > KX / 2)) {
        logCb(llvm::formatv("Unsupported padding '{0}', must be in range [0, {1}]", padLeft, KX / 2));
        return false;
    }
    if (padRight < 0 || (padRight > 1 && padRight > KX / 2)) {
        logCb(llvm::formatv("Unsupported padding '{0}', must be in range [0, {1}]", padRight, KX / 2));
        return false;
    }

    return true;
}

//
// Activation type checks
//

int64_t vpux::VPU::NCEInvariant::getAlignment(mlir::Type elemType) {
    const auto typeSizeInBits = static_cast<Bit>(getElemTypeSize(elemType));
    return std::max<int64_t>(128 / typeSizeInBits.count(), 16);
}

bool vpux::VPU::NCEInvariant::isActTypeSupported(mlir::ShapedType type, LogCb logCb) {
    if (type.getRank() != 4) {
        logCb(llvm::formatv("Activation has unsupported rank: {0}", type.getRank()));
        return false;
    }

    const auto shape = getShape(type);
    const auto order = DimsOrder::fromType(type);
    const auto memShape = order.toMemoryOrder(shape);

    const auto innerDim = memShape.back();
    const auto alignement = getAlignment(type.getElementType());

    if (innerDim % alignement != 0) {
        logCb(llvm::formatv("Activation inner dimension '{0}' is not aligned to '{1}'", innerDim, alignement));
        return false;
    }

    return true;
}

//
// PostOp checks
//

bool vpux::VPU::NCEInvariant::isPostOpSupported(mlir::Operation* postOp) {
    if (!mlir::isa<IE::ScaleShiftOp, IE::ReLUOp, IE::ClampOp, IE::SigmoidOp, IE::TanhOp>(postOp)) {
        return false;
    }

    if (auto clampOp = mlir::dyn_cast<IE::ClampOp>(postOp)) {
        const auto minVal = clampOp.minAttr().getValueAsDouble();
        if (!isDoubleEqual(minVal, 0.0)) {
            return false;
        }

        // TODO: should be check maxVal?
    }

    return true;
}

//
// WeightsTable information
//

Byte vpux::VPU::NCEInvariant::getWeightsTableSize(int64_t OC) {
    return OC * WEIGHT_TABLE_NUM_ELEMENTS_PER_OC * 4_Byte;
}