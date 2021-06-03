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

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Operation.h>

using namespace vpux;
using namespace VPUIP;

namespace {

class CMXInvariant final {
public:
    explicit CMXInvariant(Logger log = Logger::global()): _log(log.nest()) {
        _log.setName("CMXInvariant");
    }

public:
    mlir::LogicalResult verifyOp(IERT::MaxPoolOp origOp);
    mlir::LogicalResult verifyOp(IERT::ConvolutionOp origOp);

private:
    Logger _log;

private:
};

class ChannelInvariant final {
public:
    explicit ChannelInvariant(Logger log = Logger::global()): _log(log.nest()) {
        _log.setName("CMXInvariant");
    }

public:
    mlir::LogicalResult verifyOp(IERT::MaxPoolOp origOp);
    mlir::LogicalResult verifyOp(IERT::ConvolutionOp origOp);

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult verifyConcreteOp(ConcreteOp originOp, Logger log) {
    CMXInvariant cmxInvariant{log};
    if (cmxInvariant.verifyOp(originOp).failed()) {
        return mlir::failure();
    }

    ChannelInvariant channelInvariant{log};
    if (channelInvariant.verifyOp(originOp).failed()) {
        return mlir::failure();
    }

    return mlir::success();
}

//
// CMXInvariant
//

Byte getCMXSize(mlir::Operation* op) {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto resOp = IERT::RunTimeResourcesOp::getFromModule(module);

    auto cmxAttr =
            resOp.getAvailableMemory(VPUIP::PhysicalMemoryAttr::get(op->getContext(), VPUIP::PhysicalMemory::CMX_NN));
    VPUX_THROW_UNLESS(cmxAttr != nullptr, "Can't get information about {0} memory", VPUIP::PhysicalMemory::CMX_NN);

    return cmxAttr.size();
}

Byte getRequiredCMX(ArrayRef<mlir::Value> operands, int64_t numChannels) {
    Byte requiredCMX(0);

    for (const auto& operand : operands) {
        requiredCMX += getTotalSize(operand);
    }

    requiredCMX += numChannels * NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC * 4_Byte;

    return requiredCMX;
}

mlir::LogicalResult CMXInvariant::verifyOp(IERT::MaxPoolOp origOp) {
    const auto cmxSize = getCMXSize(origOp);
    const auto origInputType = origOp.input().getType().cast<mlir::MemRefType>();
    const auto inputShape = getShape(origInputType);
    const auto kernelSize = parseIntArrayAttr(origOp.kernel_size());
    const auto kernelStrides = parseIntArrayAttr(origOp.strides());

    VPUX_THROW_UNLESS(kernelSize.size() == 2, "Unsupported kernel size: %d", kernelSize.size());
    VPUX_THROW_UNLESS(kernelStrides.size() == 2, "Unsupported strides size: %d", kernelSize.size());

    if (kernelSize[0] > 11 || kernelSize[1] > 11) {
        _log.warning("{0}: Unsupported kernel size {1}x{2}. Supported size up to 11", origOp->getName(), kernelSize[0],
                     kernelSize[1]);
        return mlir::failure();
    }

    const auto IC = inputShape[IERT::MaxPoolOp::act_channel_dim()];
    const auto activationWindowSize = VPUIP::NCESparsity::getActivationWindowSize(kernelSize, kernelStrides[0],
                                                                                  origInputType.getElementType(), IC);

    auto requiredCMX = getRequiredCMX({origOp.input(), origOp.output()}, IC);
    requiredCMX += activationWindowSize * 1_Byte;

    if (requiredCMX > cmxSize) {
        _log.warning("{0}: CMX memory is not enough, available '{1}', required '{1}'", origOp->getName(), cmxSize,
                     requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult CMXInvariant::verifyOp(IERT::ConvolutionOp origOp) {
    const auto cmxSize = getCMXSize(origOp);
    const auto filterShape = getShape(origOp.filter());

    const auto OC = filterShape[IERT::ConvolutionOp::filter_out_channel_dim()];

    const auto requiredCMX = getRequiredCMX({origOp.input(), origOp.filter(), origOp.output()}, OC);
    if (requiredCMX > cmxSize) {
        _log.warning("{0}: CMX memory is not enough, available '{1}', required '{2}'", origOp->getName(), cmxSize,
                     requiredCMX);
        return mlir::failure();
    }

    return mlir::success();
}

//
// ChannelInvariant
//

mlir::LogicalResult ChannelInvariant::verifyOp(IERT::MaxPoolOp origOp) {
    const auto origInputType = origOp.input().getType().cast<mlir::MemRefType>();
    const auto inputShape = getShape(origInputType);

    const auto IC = inputShape[IERT::MaxPoolOp::act_channel_dim()];

    const auto inputType = origOp.input().getType();
    const Bit typeSizeInBits = getElemTypeSize(inputType);
    const int64_t CHANNEL_ALIGNMENT = 128 / typeSizeInBits.count();

    if (IC % CHANNEL_ALIGNMENT != 0) {
        _log.warning("{0}: Input channels are not aligned", origOp->getName());
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult ChannelInvariant::verifyOp(IERT::ConvolutionOp origOp) {
    const auto filterShape = getShape(origOp.filter());

    const auto OC = filterShape[IERT::ConvolutionOp::filter_out_channel_dim()];
    const auto IC = filterShape[IERT::ConvolutionOp::filter_in_channel_dim()];

    const auto filterType = origOp.filter().getType();
    const Bit typeSizeInBits = getElemTypeSize(filterType);
    const int64_t CHANNEL_ALIGNMENT = 128 / typeSizeInBits.count();

    if (OC % CHANNEL_ALIGNMENT != 0) {
        _log.warning("{0}: Output channels are not aligned", origOp->getName());
        return mlir::failure();
    }
    if (IC % CHANNEL_ALIGNMENT != 0) {
        _log.warning("{0}: Input channels are not aligned", origOp->getName());
        return mlir::failure();
    }

    return mlir::success();
}

}  // namespace

//
// NCEInvariant
//

mlir::LogicalResult vpux::VPUIP::NCEInvariant::verifyOp(mlir::Operation* op, Logger log) {
    return llvm::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(op)
            .Case<IERT::MaxPoolOp>([&](IERT::MaxPoolOp originOp) {
                return verifyConcreteOp(originOp, log);
            })
            .Case<IERT::ConvolutionOp>([&](IERT::ConvolutionOp originOp) {
                return verifyConcreteOp(originOp, log);
            })
            .Default([](mlir::Operation* unknownOp) -> mlir::LogicalResult {
                VPUX_THROW("Operation '{0}' the operation is not supported by the DPU", unknownOp->getName());
            });
}