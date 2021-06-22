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

int64_t vpux::VPUIP::NCEInvariant::getChannelAlignment(mlir::Type elemType) {
    const Bit typeSizeInBits = getElemTypeSize(elemType);
    return std::max<int64_t>(128 / typeSizeInBits.count(), 16);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyConvChannels(mlir::Location loc, mlir::ShapedType filterType,
                                                                  Logger log) {
    log.setName("NCEInvariant");

    const auto filterShape = getShape(filterType);
    const auto OC = filterShape[IERT::ConvolutionOp::filter_out_channel_dim()];
    const auto IC = filterShape[IERT::ConvolutionOp::filter_in_channel_dim()];

    if (OC % getChannelAlignment(filterType.getElementType()) != 0) {
        log.trace("[{0}] Convolution output channels are not aligned", loc);
        return mlir::failure();
    }
    if (IC % getChannelAlignment(filterType.getElementType()) != 0) {
        log.trace("[{0}] Convolution input channels are not aligned", loc);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IE::ConvolutionOp origOp, Logger log) {
    return verifyConvChannels(origOp->getLoc(), origOp.filter().getType().cast<mlir::ShapedType>(), log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IERT::ConvolutionOp origOp, Logger log) {
    return verifyConvChannels(origOp->getLoc(), origOp.filter().getType().cast<mlir::ShapedType>(), log);
}

//
// verifyPoolChannels
//

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyPoolChannels(mlir::Location loc, mlir::ShapedType inputType,
                                                                  Logger log) {
    log.setName("NCEInvariant");

    const auto inputShape = getShape(inputType);
    const auto IC = inputShape[IERT::MaxPoolOp::act_channel_dim()];

    if (IC % getChannelAlignment(inputType.getElementType()) != 0) {
        log.trace("[{0}] Pooling channels are not aligned", loc);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IE::MaxPoolOp origOp, Logger log) {
    return verifyPoolChannels(origOp->getLoc(), origOp.input().getType().cast<mlir::ShapedType>(), log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyChannels(IERT::MaxPoolOp origOp, Logger log) {
    return verifyPoolChannels(origOp->getLoc(), origOp.input().getType().cast<mlir::ShapedType>(), log);
}

//
// verifyConvCMX
//

namespace {

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

}  // namespace

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyConvCMX(mlir::Location loc, mlir::ModuleOp module,
                                                             mlir::MemRefType inputType, mlir::MemRefType filterType,
                                                             mlir::MemRefType outputType, Logger log) {
    log.setName("NCEInvariant");

    const auto filterShape = getShape(filterType);
    const auto OC = filterShape[IERT::ConvolutionOp::filter_out_channel_dim()];

    const auto requiredCMX = getRequiredCMX({inputType, filterType, outputType}, OC);

    const auto cmxSize = getCMXSize(module);
    if (requiredCMX > cmxSize) {
        log.trace("[{0}] CMX memory is not enough for Convolution, available '{1}', required '{2}'", loc, cmxSize,
                  requiredCMX);
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
    log.setName("NCEInvariant");

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
        log.trace("[{0}] CMX memory is not enough for Pooling, available '{1}', required '{2}'", loc, cmxSize,
                  requiredCMX);
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

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(mlir::Location loc, mlir::ArrayAttr kernelSizeAttr,
                                                            mlir::ArrayAttr kernelStridesAttr, Logger log) {
    log.setName("NCEInvariant");

    const auto kernelSize = parseIntArrayAttr(kernelSizeAttr);
    const auto KY = kernelSize[0];
    const auto KX = kernelSize[1];

    static const int32_t NCE_MAX_KERNEL_SIZE = 11;

    if (KY > NCE_MAX_KERNEL_SIZE || KY <= 0) {
        log.trace("[{0}] Unsupported kernel height dimension '{1}', must be in range [1, {2}]", loc, KY,
                  NCE_MAX_KERNEL_SIZE);
        return mlir::failure();
    }
    if (KX > NCE_MAX_KERNEL_SIZE || KX <= 0) {
        log.trace("[{0}] Unsupported kernel width dimension '{1}', must be in range [1, {2}]", loc, KX,
                  NCE_MAX_KERNEL_SIZE);
        return mlir::failure();
    }

    const auto kernelStrides = parseIntArrayAttr(kernelStridesAttr);
    const auto SY = kernelStrides[0];
    const auto SX = kernelStrides[1];

    static const int32_t NCE_MAX_STRIDE_SIZE = 8;

    if (SX != SY) {
        log.trace("[{0}] Assymetric strides are not supported", loc);
        return mlir::failure();
    }
    if (SY > NCE_MAX_STRIDE_SIZE || SY <= 0) {
        log.trace("[{0}] Unsupported stride height dimension '{1}', must be in range [1, {2}]", loc, SY,
                  NCE_MAX_STRIDE_SIZE);
        return mlir::failure();
    }
    if (SX > NCE_MAX_STRIDE_SIZE || SX <= 0) {
        log.trace("[{0}] Unsupported stride width dimension '{1}', must be in range [1, {2}]", loc, SX,
                  NCE_MAX_STRIDE_SIZE);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(IE::ConvolutionOp origOp, Logger log) {
    log.setName("NCEInvariant");

    const auto dilations = parseIntArrayAttr(origOp.dilations());
    if (dilations[0] != 1 || dilations[1] != 1) {
        log.trace("[{0}] Unsupported kernel dilations '{1}'", origOp->getLoc(), dilations);
        return mlir::failure();
    }

    const auto filterShape = getShape(origOp.filter());
    const auto KY = filterShape[IERT::ConvolutionOp::filter_spatial_height_dim()];
    const auto KX = filterShape[IERT::ConvolutionOp::filter_spatial_width_dim()];
    const auto kernelSizeAttr = getInt32ArrayAttr(origOp.getContext(), makeArrayRef({KY, KX}));

    return verifyKernel(origOp->getLoc(), kernelSizeAttr, origOp.strides(), log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(IERT::ConvolutionOp origOp, Logger log) {
    log.setName("NCEInvariant");

    const auto dilations = parseIntArrayAttr(origOp.dilations());
    if (dilations[0] != 1 || dilations[1] != 1) {
        log.trace("[{0}] Unsupported kernel dilations '{1}'", origOp->getLoc(), dilations);
        return mlir::failure();
    }

    const auto filterShape = getShape(origOp.filter());
    const auto KY = filterShape[IERT::ConvolutionOp::filter_spatial_height_dim()];
    const auto KX = filterShape[IERT::ConvolutionOp::filter_spatial_width_dim()];
    const auto kernelSizeAttr = getInt32ArrayAttr(origOp.getContext(), makeArrayRef({KY, KX}));

    return verifyKernel(origOp->getLoc(), kernelSizeAttr, origOp.strides(), log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(IE::MaxPoolOp origOp, Logger log) {
    log.setName("NCEInvariant");

    const auto kernelSize = parseIntArrayAttr(origOp.kernel_size());
    if (kernelSize[0] != kernelSize[1]) {
        log.trace("[{0}] Assymetric kernel is not supported", origOp->getLoc());
        return mlir::failure();
    }

    return verifyKernel(origOp->getLoc(), origOp.kernel_size(), origOp.strides(), log);
}

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyKernel(IERT::MaxPoolOp origOp, Logger log) {
    log.setName("NCEInvariant");

    const auto kernelSize = parseIntArrayAttr(origOp.kernel_size());
    if (kernelSize[0] != kernelSize[1]) {
        log.trace("[{0}] Assymetric kernel is not supported", origOp->getLoc());
        return mlir::failure();
    }

    return verifyKernel(origOp->getLoc(), origOp.kernel_size(), origOp.strides(), log);
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
                VPUX_THROW("Operation '{0}' at '{1}' is not supported by the NCE", unknownOp->getName(),
                           unknownOp->getLoc());
            });
}
