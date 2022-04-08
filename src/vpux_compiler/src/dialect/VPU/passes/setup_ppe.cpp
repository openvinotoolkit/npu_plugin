//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

VPU::PPEMode setupActivationFunction(const VPU::PPEModeAttr ppeModeAttr, const int32_t clampLow,
                                     const int32_t clampHigh, const int32_t lreluShift) {
    if (ppeModeAttr != nullptr && lreluShift != 0) {
        return VPU::PPEMode::LPRELU;
    } else {
        if ((clampHigh == std::numeric_limits<int32_t>::max() || clampHigh == 0) && clampLow == 0) {
            return VPU::PPEMode::LRELU;
        } else if (clampHigh < std::numeric_limits<int32_t>::max() && clampLow == 0) {
            return VPU::PPEMode::LRELUX;
        } else {
            return VPU::PPEMode::NOOP;
        }
    }

    return VPU::PPEMode::NOOP;
}

int32_t setupClampLow(const mlir::Type elemType) {
    const auto outElemQType = elemType.dyn_cast<mlir::quant::QuantizedType>();
    if (outElemQType == nullptr) {
        return std::numeric_limits<int32_t>::min();
    }
    return checked_cast<int32_t>(outElemQType.getStorageTypeMin());
}

int32_t fixedPointToHalf(const int64_t fixedPoint) {
    const float clampF32 = checked_cast<float>(fixedPoint) / (1 << 16);
    const ngraph::float16 half(clampF32);
    const int16_t* ptrI16 = reinterpret_cast<const int16_t*>(&half);
    const int16_t i16 = *ptrI16;
    return i16;
}

int32_t fixedPointToBFloat(const int64_t fixedPoint) {
    const float clampF32 = checked_cast<float>(fixedPoint) / (1 << 16);
    const ngraph::bfloat16 biFloat(clampF32);
    const int16_t* ptrI16 = reinterpret_cast<const int16_t*>(&biFloat);
    const int16_t i16 = *ptrI16;
    return i16;
}

int32_t setupClampHigh(const mlir::Type elemType, const bool isReLUX, const int32_t oldClampHigh = 0) {
    if (isReLUX) {
        if (elemType.isF16()) {
            return fixedPointToHalf(oldClampHigh);
        } else if (elemType.isBF16()) {
            return fixedPointToBFloat(oldClampHigh);
        } else {
            return oldClampHigh;
        }
    }

    const auto outElemQType = elemType.dyn_cast<mlir::quant::QuantizedType>();
    if (outElemQType == nullptr) {
        return std::numeric_limits<int32_t>::max();
    }
    return checked_cast<int32_t>(outElemQType.getStorageTypeMax());
}

float readReluMult(const int32_t Lrelu_Mult) {
    const int32_t LRELU_MULT_MASK = 0x7FF;
    // LRELU_MULT_MASK must not consider the sign
    if (Lrelu_Mult >= 0) {
        VPUX_THROW_WHEN(Lrelu_Mult & ~LRELU_MULT_MASK, "ReLU multiplier exceeds 11 bits");
    } else {
        VPUX_THROW_WHEN(-Lrelu_Mult & ~LRELU_MULT_MASK, "ReLU multiplier exceeds 11 bits");
    }
    return checked_cast<float>(Lrelu_Mult);
}

int32_t readReluShift(const int32_t Lrelu_Shift) {
    const int32_t LRELU_SHIFT_MASK = 0x1F;
    VPUX_THROW_WHEN(Lrelu_Shift & ~LRELU_SHIFT_MASK, "ReLU shift does not match the expected format");
    return Lrelu_Shift;
}

float calculateFpPReluAlpha(const VPU::PPEMode& ppeType, const int32_t LreluMult, const int32_t LreluShift) {
    // Only applicable for LPRELU mode
    if (ppeType != VPU::PPEMode::LPRELU) {
        return 1.f;
    }
    const auto lReluMult = readReluMult(LreluMult);
    const auto lReluShift = readReluShift(LreluShift);
    return std::ldexp(lReluMult, -lReluShift);
}

float rescaleNegativeSlope(const float preluMult, const mlir::Type inType, const mlir::Type outType) {
    // Mixed mode with float input and quant output requires negative slope rescaling.
    if (!inType.isa<mlir::quant::QuantizedType>() && outType.isa<mlir::quant::UniformQuantizedType>()) {
        const auto perTensor = outType.cast<mlir::quant::UniformQuantizedType>();
        return preluMult / static_cast<float>(perTensor.getScale());
    }

    return preluMult;
}

void applyPpeRewriter(VPU::NCEOpInterface origOp, Logger log) {
    log.setName("applyPpeRewriter");
    log.trace("Got operation '{1}' at '{2}'", origOp->getName(), origOp->getLoc());

    int32_t clampLow = std::numeric_limits<int32_t>::min();
    int32_t clampHigh = std::numeric_limits<int32_t>::max();
    int32_t lreluMult = 1;
    int32_t lreluShift = 0;

    mlir::IntegerAttr clampLowAttr = nullptr;
    mlir::IntegerAttr clampHighAttr = nullptr;
    mlir::IntegerAttr lreluMultAttr = nullptr;
    mlir::IntegerAttr lreluShiftAttr = nullptr;
    mlir::ArrayAttr quantScaleAttr = nullptr;
    mlir::ArrayAttr quantMultAttr = nullptr;
    mlir::ArrayAttr quantShiftAttr = nullptr;
    mlir::IntegerAttr quantPostShiftAttr = nullptr;
    VPU::PPEModeAttr ppeModeAttr = nullptr;
    mlir::ArrayAttr lhsQuantMult = nullptr;
    mlir::ArrayAttr rhsQuantMult = nullptr;

    const auto maybePpeTask = origOp.getPPE();
    if (maybePpeTask.hasValue()) {
        const auto ppeTask = maybePpeTask.getValue();
        ppeModeAttr = ppeTask.mode();

        clampLowAttr = ppeTask.clamp_low();
        clampHighAttr = ppeTask.clamp_high();
        lreluMultAttr = ppeTask.lrelu_mult();
        lreluShiftAttr = ppeTask.lrelu_shift();

        quantScaleAttr = ppeTask.quant_scale();
        quantMultAttr = ppeTask.quant_mult();
        quantShiftAttr = ppeTask.quant_shift();
        quantPostShiftAttr = ppeTask.quant_post_shift();
        lhsQuantMult = ppeTask.in1_quant_mult();
        rhsQuantMult = ppeTask.in2_quant_mult();

        clampLow = checked_cast<int32_t>(clampLowAttr.getValue().getSExtValue());
        clampHigh = checked_cast<int32_t>(clampHighAttr.getValue().getSExtValue());
        lreluMult = checked_cast<int32_t>(lreluMultAttr.getValue().getSExtValue());
        lreluShift = checked_cast<int32_t>(lreluShiftAttr.getValue().getSExtValue());
    }

    const auto ppeMode = setupActivationFunction(ppeModeAttr, clampLow, clampHigh, lreluShift);
    log.nest().trace("setup activation function '{0}' for '{1}'", ppeMode, origOp->getLoc());

    int64_t newClampLow = 0;
    int64_t newClampHigh = 0;

    const auto outElemType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getElementType();
    // Consider NOOP as a Clamp
    auto isNOOP = ppeMode == VPU::PPEMode::NOOP;
    if (isNOOP && outElemType.isa<mlir::quant::QuantizedType>()) {
        if (ppeModeAttr == nullptr) {
            newClampLow = setupClampLow(outElemType);
            newClampHigh = setupClampHigh(outElemType, false);
        } else {
            const auto zps = extractScalesAndZeroPoints(outElemType).second;
            newClampLow = clampLow + zps.front();
            newClampHigh = clampHigh + zps.front();
        }
    } else {
        if (!outElemType.isa<mlir::quant::QuantizedType>()) {
            // More information here: #50875
            VPUX_THROW_WHEN(clampLow != std::numeric_limits<int32_t>::min() && clampLow != 0,
                            "Unsupported clamp low value: {0}", clampLow);
        }
        newClampLow = setupClampLow(outElemType);

        const bool isReLUX = (ppeMode == VPU::PPEMode::LRELUX);
        newClampHigh = setupClampHigh(outElemType, isReLUX, clampHigh);
    }

    log.nest().trace("setup clamping '{0}'-'{1}' for '{2}'", newClampLow, newClampHigh, origOp->getLoc());

    const float fpPReluAlpha = calculateFpPReluAlpha(ppeMode, lreluMult, lreluShift);
    const auto inElemType = origOp->getOperand(0).getType().cast<vpux::NDTypeInterface>().getElementType();
    const float rescaledAlpha = rescaleNegativeSlope(fpPReluAlpha, inElemType, outElemType);

    auto ctx = origOp.getContext();
    auto ppeTaskAttr = VPU::PPETaskAttr::get(
            VPU::PPEModeAttr::get(ctx, ppeMode), getIntAttr(ctx, newClampLow), getIntAttr(ctx, newClampHigh),
            getIntAttr(ctx, lreluMult), getIntAttr(ctx, lreluShift), quantScaleAttr, quantMultAttr, quantShiftAttr,
            quantPostShiftAttr, lhsQuantMult, rhsQuantMult, getFPAttr(ctx, rescaledAlpha), ctx);

    origOp.setPPE(ppeTaskAttr);
}

//
// SetupPPEPass
//

class SetupPPEPass final : public VPU::SetupPPEPassBase<SetupPPEPass> {
public:
    explicit SetupPPEPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void SetupPPEPass::safeRunOnFunc() {
    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    const std::set<VPU::ArchKind> compatibleTargets = {
            VPU::ArchKind::VPUX37XX,
    };
    if (compatibleTargets.count(arch) == 0) {
        _log.trace("SetupPPEPass is only applicable for VPUX37XX device.");
        return;
    }

    func->walk([&](VPU::NCEOpInterface op) {
        applyPpeRewriter(op, _log);
    });
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createSetupPPEPass(Logger log) {
    return std::make_unique<SetupPPEPass>(log);
}
