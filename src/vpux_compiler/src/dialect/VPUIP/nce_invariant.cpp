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

#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_sparsity.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Operation.h>

using namespace vpux;
using namespace VPUIP;

//
// verifyConvChannels
//

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyConvChannels(mlir::Location loc, mlir::MemRefType filterType,
                                                                  Logger log) {
    log.setName("NCEInvariant::verifyConvChannels");

    const auto filterShape = getShape(filterType);

    const auto OC = filterShape[IERT::ConvolutionOp::filter_out_channel_dim()];
    const auto IC = filterShape[IERT::ConvolutionOp::filter_in_channel_dim()];

    const Bit typeSizeInBits = getElemTypeSize(filterType);
    const int64_t CHANNEL_ALIGNMENT = 128 / typeSizeInBits.count();

    if (OC % CHANNEL_ALIGNMENT != 0) {
        log.warning("{0}: Output channels are not aligned", loc);
        return mlir::failure();
    }
    if (IC % CHANNEL_ALIGNMENT != 0) {
        log.warning("{0}: Input channels are not aligned", loc);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IERT::ConvolutionOp origOp, Logger log) {
    return verifyConvChannels(origOp->getLoc(), origOp.filter().getType().cast<mlir::MemRefType>(), log);
}

//
// verifyPoolChannels
//

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPoolChannels(mlir::Location loc, mlir::MemRefType inputType,
                                                                  Logger log) {
    log.setName("NCEInvariant::verifyPoolChannels");

    const auto inputShape = getShape(inputType);

    const auto IC = inputShape[IERT::MaxPoolOp::act_channel_dim()];

    const Bit typeSizeInBits = getElemTypeSize(inputType);
    const int64_t CHANNEL_ALIGNMENT = 128 / typeSizeInBits.count();

    if (IC % CHANNEL_ALIGNMENT != 0) {
        log.warning("{0}: Input channels are not aligned", loc);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IERT::MaxPoolOp origOp, Logger log) {
    return verifyPoolChannels(origOp->getLoc(), origOp.input().getType().cast<mlir::MemRefType>(), log);
}

//
// verifyConvCMX
//

Byte getCMXSize(mlir::ModuleOp module) {
    auto resOp = IERT::RunTimeResourcesOp::getFromModule(module);

    const auto cmxAttr = VPUIP::PhysicalMemoryAttr::get(module->getContext(), VPUIP::PhysicalMemory::CMX_NN);

    auto cmxRes = resOp.getAvailableMemory(cmxAttr);
    VPUX_THROW_UNLESS(cmxRes != nullptr, "Can't get information about {0} memory", VPUIP::PhysicalMemory::CMX_NN);

    return cmxRes.size();
}

Byte getRequiredCMX(ArrayRef<mlir::MemRefType> operands, int64_t numChannels) {
    Byte requiredCMX(0);

    for (const auto& operand : operands) {
        requiredCMX += getTypeTotalSize(operand);
    }

    requiredCMX += numChannels * NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC * 4_Byte;

    return requiredCMX;
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyConvCMX(mlir::Location loc, mlir::ModuleOp module,
                                                             mlir::MemRefType inputType, mlir::MemRefType filterType,
                                                             mlir::MemRefType outputType, Logger log) {
    log.setName("NCEInvariant::verifyConvCMX");

    const auto filterShape = getShape(filterType);
    const auto OC = filterShape[IERT::ConvolutionOp::filter_out_channel_dim()];

    const auto requiredCMX = getRequiredCMX({inputType, filterType, outputType}, OC);

    const auto cmxSize = getCMXSize(module);
    if (requiredCMX > cmxSize) {
        log.warning("{0}: CMX memory is not enough, available '{1}', required '{2}'", loc, cmxSize, requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyCMX(IERT::ConvolutionOp origOp, Logger log) {
    return verifyConvCMX(origOp->getLoc(), origOp->getParentOfType<mlir::ModuleOp>(),
                         origOp.input().getType().cast<mlir::MemRefType>(),
                         origOp.filter().getType().cast<mlir::MemRefType>(),
                         origOp.output().getType().cast<mlir::MemRefType>(), log);
}

//
// verifyPoolCMX
//

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPoolCMX(mlir::Location loc, mlir::ModuleOp module,
                                                             mlir::MemRefType inputType, mlir::MemRefType outputType,
                                                             mlir::ArrayAttr kernelSize, mlir::ArrayAttr kernelStrides,
                                                             Logger log) {
    log.setName("NCEInvariant::verifyPoolCMX");

    VPUX_THROW_UNLESS(kernelSize.size() == 2, "Unsupported kernel size: {0}", kernelSize.size());
    VPUX_THROW_UNLESS(kernelStrides.size() == 2, "Unsupported strides size: {0}", kernelSize.size());

    const auto inputShape = getShape(inputType);
    const auto IC = inputShape[IERT::MaxPoolOp::act_channel_dim()];

    const auto kernelSizeVals = parseIntArrayAttr(kernelSize);
    const auto kernelStridesVals = parseIntArrayAttr(kernelStrides);

    const auto activationWindowSize = VPUIP::NCESparsity::getActivationWindowSize(kernelSizeVals, kernelStridesVals[0],
                                                                                  inputType.getElementType(), IC);

    const auto requiredCMX = getRequiredCMX({inputType, outputType}, IC) + activationWindowSize * 1_Byte;

    const auto cmxSize = getCMXSize(module);
    if (requiredCMX > cmxSize) {
        log.warning("{0}: CMX memory is not enough, available '{1}', required '{2}'", loc, cmxSize, requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyCMX(IERT::MaxPoolOp origOp, Logger log) {
    return verifyPoolCMX(origOp->getLoc(), origOp->getParentOfType<mlir::ModuleOp>(),
                         origOp.input().getType().cast<mlir::MemRefType>(),
                         origOp.output().getType().cast<mlir::MemRefType>(), origOp.kernel_size(), origOp.strides(),
                         log);
}

//
// verifyKernel
//

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(mlir::ArrayAttr kernelSizeAttr,
                                                            mlir::ArrayAttr kernelStridesAttr,
                                                            mlir::ArrayAttr kernelPaddingAttr, Logger log) {
    log.setName("NCEInvariant::verifyKernel");

    if (kernelSizeAttr == nullptr) {
        log.warning("kernel size attribute is required");
        return mlir::failure();
    }
    if (kernelStridesAttr == nullptr) {
        log.warning("kernel strides attribute is required");
        return mlir::failure();
    }
    if (kernelPaddingAttr == nullptr) {
        log.warning("kernel padding attribute is required");
        return mlir::failure();
    }

    const auto kernelSize = parseIntArrayAttr(kernelSizeAttr);
    const auto kernelStrides = parseIntArrayAttr(kernelStridesAttr);

    static const int32_t NCE_MAX_KERNEL_SIZE = 11;
    static const int32_t NCE_MAX_STRIDE_SIZE = 8;

    if (kernelSize.size() != 2) {
        log.warning("Unsupported kernel size: {0}", kernelSize.size());
        return mlir::failure();
    }

    const auto KY = kernelSize[0];
    const auto KX = kernelSize[1];

    if (KY > NCE_MAX_KERNEL_SIZE || KY <= 0) {
        log.warning("{0}: Unsupported kernel height dimension: '{0}'. Must be between 1-{1}.", KY, NCE_MAX_KERNEL_SIZE);
        return mlir::failure();
    }

    if (KX > NCE_MAX_KERNEL_SIZE || KX <= 0) {
        log.warning("{0}: Unsupported kernel width dimension: '{0}'. Must be between 1-{1}.", KX, NCE_MAX_KERNEL_SIZE);
        return mlir::failure();
    }

    const auto SY = kernelStrides[0];
    const auto SX = kernelStrides[1];

    if (SY > NCE_MAX_STRIDE_SIZE || SY <= 0) {
        log.warning("{0}: Unsupported stride height dimension: '{0}'. Must be between 1-{1}.", SY, NCE_MAX_STRIDE_SIZE);
        return mlir::failure();
    }

    if (SX > NCE_MAX_STRIDE_SIZE || SX <= 0) {
        log.warning("{0}: Unsupported stride width dimension: '{0}'. Must be between 1-{1}.", SX, NCE_MAX_STRIDE_SIZE);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(IERT::ConvolutionOp origOp, Logger log) {
    log.setName("NCEInvariant::verifyConvKernel");

    const auto filterShape = getShape(origOp.filter());
    const auto KY = filterShape[IERT::ConvolutionOp::filter_spatial_height_dim()];
    const auto KX = filterShape[IERT::ConvolutionOp::filter_spatial_width_dim()];
    const auto kernelSizeAttr = getInt32ArrayAttr(origOp.getContext(), makeArrayRef({KX, KY}));

    const auto padsBegin = parseIntArrayAttr(origOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr(origOp.pads_end());
    const auto kernelPaddingAttr =
            getInt32ArrayAttr(origOp.getContext(), makeArrayRef({padsBegin[1], padsEnd[1], padsBegin[0], padsEnd[0]}));

    return verifyKernel(kernelSizeAttr, origOp.strides(), kernelPaddingAttr);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(IERT::MaxPoolOp origOp, Logger log) {
    log.setName("NCEInvariant::verifyPoolKernel");

    const auto padsBegin = parseIntArrayAttr(origOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr(origOp.pads_end());
    const auto kernelPaddingAttr =
            getInt32ArrayAttr(origOp.getContext(), makeArrayRef({padsBegin[1], padsEnd[1], padsBegin[0], padsEnd[0]}));

    return verifyKernel(origOp.kernel_size(), origOp.strides(), kernelPaddingAttr);
}

//
// verifyOp
//

namespace {

template <class ConcreteOp>
mlir::LogicalResult verifyConcreteOp(ConcreteOp origOp, Logger log) {
    if (mlir::failed(VPUIP::NCEInvariant::verifyKernel(origOp, log))) {
        return mlir::failure();
    }

    if (mlir::failed(VPUIP::NCEInvariant::verifyChannels(origOp, log))) {
        return mlir::failure();
    }

    if (mlir::failed(VPUIP::NCEInvariant::verifyCMX(origOp, log))) {
        return mlir::failure();
    }

    return mlir::success();
}

}  // namespace

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyOp(mlir::Operation* op, Logger log) {
    return llvm::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(op)
            .Case<IERT::ConvolutionOp>([&](IERT::ConvolutionOp origOp) {
                return verifyConcreteOp(origOp, log);
            })
            .Case<IERT::MaxPoolOp>([&](IERT::MaxPoolOp origOp) {
                return verifyConcreteOp(origOp, log);
            })
            .Default([](mlir::Operation* unknownOp) -> mlir::LogicalResult {
                VPUX_THROW("Operation '{0}' is not supported by the NCE", unknownOp->getName());
            });
}
