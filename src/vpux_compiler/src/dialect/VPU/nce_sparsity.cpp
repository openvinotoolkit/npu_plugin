//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/IE/loop.hpp"
#include "vpux/utils/core/algo.hpp"
#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <limits>
#include <numeric>

using namespace vpux;

namespace {

using namespace VPU::NCESparsity;

bool isMixedPrecisionSupported(VPU::ArchKind arch) {
    return arch != VPU::ArchKind::VPUX30XX && arch != VPU::ArchKind::VPUX311X;
}

Scales exractWeightsScales(mlir::Type weightsElemType) {
    if (weightsElemType == nullptr || !weightsElemType.isa<mlir::quant::QuantizedType>()) {
        return SmallVector<double>{1.0};
    }

    return extractScalesAndZeroPoints(weightsElemType).first;
}

mlir::Type getBaseStorageType(mlir::Type elemType) {
    if (auto quant = elemType.dyn_cast_or_null<mlir::quant::QuantizedType>()) {
        return quant.getStorageType();
    }

    return elemType;
}

int64_t getWindowSize(int64_t KX, int64_t SX, mlir::Type elemType) {
    VPUX_THROW_UNLESS(KX <= 11, "Unsupported kernel size {0}. Supported size up to 11", KX);

    // Select the maximum window size not exceeding 32 bytes
    // by iterating through the MPE_NUM values (2, 4, 8, 16)

    auto actualType = getBaseStorageType(elemType);
    VPUX_THROW_UNLESS(actualType.isInteger(CHAR_BIT) || actualType.isF16() || actualType.isBF16(),
                      "Supported only U8/I8 and FP16/BF16 types {0}", actualType);

    // Only MPE0, MPE4, MPE8 and MPE12 support FP16 data format
    const int mpeNumLimit = actualType.isF16() ? 4 : 16;

    const Bit typeSizeInBits = getElemTypeSize(actualType);

    // Window size is limited to 32 bytes by HW. Size of the data type
    // needs to be accounted to find the max (32 for U8, 16 for FP16)
    const int64_t maxWindowSize = 32 / (typeSizeInBits.count() / 8);
    int64_t maxMpeWindowSize = 64;

    int64_t windowSize = 0;
    int mpeNum = 1;

    while (mpeNum <= mpeNumLimit) {
        if (SX <= KX) {
            windowSize = KX + SX * (mpeNum - 1);
        } else {
            windowSize = KX * mpeNum;
        }

        if (windowSize <= maxWindowSize)
            maxMpeWindowSize = windowSize;

        mpeNum *= 2;
    }

    return maxMpeWindowSize;
}

std::vector<uint8_t> getBitPattern(VPU::NCESparsity::Mode mode, ShapeRef kernelSize, int64_t windowSize, int64_t IC) {
    const auto KY = kernelSize[Dims4D::Kernel::Y];
    const auto KX = kernelSize[Dims4D::Kernel::X];

    VPUX_THROW_UNLESS(windowSize >= KX, "windowsSize must be greater than or equal to KX. windowsSize={0}, KX={1}",
                      windowSize, KX);

    const auto numBitsSet = KX;
    const auto numBitsClear = windowSize - KX;

    SmallVector<uint8_t> window;
    window.reserve(windowSize);
    window.insert(window.end(), numBitsSet, 1);
    window.insert(window.end(), numBitsClear, 0);

    const auto numOfRepeat = mode == VPU::NCESparsity::Mode::CM_CONV ? KY * IC : KY;

    std::vector<uint8_t> bitPattern;
    bitPattern.reserve(numOfRepeat * windowSize);
    for (auto i = 0; i < numOfRepeat; i++) {
        bitPattern.insert(bitPattern.end(), window.begin(), window.end());
    }

    return bitPattern;
}

constexpr std::int32_t ALIGNMENT_REQUIREMENT_IN_ELEMENTS = 16;

constexpr int32_t getVPUX30XX_Scale(uint8_t shift, uint16_t mult, double, mlir::Type) {
    // FIXME: set value when PPE is LPRELU in quant mode
    int32_t PRELU_SCALE_OFFSET = 0;
    int32_t PRELU_SCALE_VALUE = 1;

    int32_t PPE_SHIFT_OFFSET = 8;
    int32_t PPE_SHIFT_VALUE = shift;

    int32_t ROUND_MODE_OFFSET = 14;
    int32_t ROUND_MODE_VALUE = 1;

    int32_t PPE_MULT_OFFSET = 16;
    // FIXME: PPE multiplier has sign, which may affect lower bits
    int32_t PPE_MULT_VALUE = mult;

    return (PRELU_SCALE_VALUE << PRELU_SCALE_OFFSET) | (PPE_SHIFT_VALUE << PPE_SHIFT_OFFSET) |
           (ROUND_MODE_VALUE << ROUND_MODE_OFFSET) | (PPE_MULT_VALUE << PPE_MULT_OFFSET);
}

int32_t getVPUX37XX_Scale(uint8_t shift, uint16_t mult, double rescale, mlir::Type inputType) {
    // VPUX37XX expects scale in IEEE754 format in NCE_DPU_PPE_FP_SCALE register in case input has FP16/BF16 type
    if (inputType.isa<mlir::FloatType>()) {
        return VPU::NCESparsity::toHex(rescale);
    }

    int32_t PPE_SHIFT_OFFSET = 8;
    int32_t PPE_SHIFT_VALUE = shift;

    int32_t PPE_MULT_OFFSET = 16;
    // FIXME: PPE multiplier has sign, which may affect lower bits
    int32_t PPE_MULT_VALUE = mult;

    return (PPE_SHIFT_VALUE << PPE_SHIFT_OFFSET) | (PPE_MULT_VALUE << PPE_MULT_OFFSET);
}

llvm::unique_function<int32_t(size_t)> getBiasFunc(mlir::Type inElemType, mlir::Type outElemType,
                                                   mlir::Type weightsElemType, Const::ContentAttr bias,
                                                   VPU::ArchKind arch, size_t OC) {
    if (bias == nullptr) {
        return [](int64_t) -> double {
            return 0.0f;
        };
    }

    auto biasContent = bias.fold();

    const auto isInQuantized = inElemType.isa<mlir::quant::QuantizedType>();
    const auto isOutQuantized = outElemType.isa<mlir::quant::QuantizedType>();
    const auto isQuant = isInQuantized && isOutQuantized;
    const auto isFloat = !isInQuantized && !isOutQuantized;
    const auto isMixedModeSupported = isMixedPrecisionSupported(arch);
    const auto isMixed = !isQuant && !isFloat && isMixedModeSupported;
    const auto isQuantInFloatOut = isInQuantized && isMixed;
    const auto isFloatInQuantOut = isOutQuantized && isMixed;
    if (isQuant || isQuantInFloatOut) {
        // PPE engages float by-pass in this case. Apply re-scaling.
        auto rescaledBias = VPU::NCESparsity::getRescaledBias(bias, inElemType, weightsElemType, OC);
        VPUX_THROW_WHEN(mlir::failed(rescaledBias), "Rescaled bias value is out of range");

        return [rescaledBiasValue = std::move(rescaledBias.getValue())](size_t oc) -> int32_t {
            return checked_cast<int32_t>(std::round(rescaledBiasValue[oc]));
        };
    } else if (isFloat || isFloatInQuantOut) {
        return [biasContent = std::move(biasContent), arch](int64_t oc) -> int32_t {
            const auto biasVal = biasContent.getValues<float>()[oc];
            const auto biasConverter = VPU::NCESparsity::biasConvertersMap.at(arch);
            return biasConverter(biasVal);
        };
    }

    if (isMixedModeSupported) {
        VPUX_THROW("In/Out element type of NCE op mismatch. quant-quant, quant-float, float-quant or float-float type "
                   "pairs required. Got: in type {0}, out type {1}",
                   inElemType, outElemType);
    } else {
        VPUX_THROW("In/Out element type of NCE op mismatch. Both types must be quantized or not quantized. Got: in "
                   "type {0}, out type {1}",
                   inElemType, outElemType);
    }
}

llvm::unique_function<int32_t(size_t)> getMultShiftFunc(mlir::Type inElemType, mlir::Type outElemType,
                                                        mlir::Type weightsType, VPU::PPETaskAttr ppe,
                                                        VPU::ArchKind arch, size_t OC, bool isSpecialRelu) {
    auto changeMultForPPE = [](int32_t& mult, VPU::PPETaskAttr ppe) {
        if (ppe && ppe.mode().getValue() == VPU::PPEMode::LPRELU) {
            mult &= 0xFFFFFF00;
            mult |= static_cast<int32_t>(ppe.lrelu_mult().getInt());
        }
    };
    auto bypassMultForPPE = [](int32_t&, VPU::PPETaskAttr) {};
    auto updateMultForPPE =
            (arch == VPU::ArchKind::VPUX30XX || arch == VPU::ArchKind::VPUX311X) ? changeMultForPPE : bypassMultForPPE;
    if (!inElemType.isa<mlir::quant::QuantizedType>() && !outElemType.isa<mlir::quant::QuantizedType>()) {
        const auto ppeConverter = VPU::NCESparsity::ppeConvertersMap.at(arch);
        int32_t multShift = ppeConverter(0, 1, 1, inElemType);
        updateMultForPPE(multShift, ppe);
        return [multShift](size_t) {
            return multShift;
        };
    } else if ((inElemType.isa<mlir::quant::QuantizedType>() && outElemType.isa<mlir::quant::QuantizedType>()) ||
               isMixedPrecisionSupported(arch)) {
        auto inQuantScale = inElemType.isa<mlir::quant::QuantizedType>() ? extractScalesAndZeroPoints(inElemType).first
                                                                         : SmallVector<double>{1.0};
        auto outQuantScale = outElemType.isa<mlir::quant::QuantizedType>()
                                     ? extractScalesAndZeroPoints(outElemType).first
                                     : SmallVector<double>{1.0};
        auto weightsQuantScales = exractWeightsScales(weightsType);

        broadcast(inQuantScale, OC);
        broadcast(outQuantScale, OC);
        broadcast(weightsQuantScales, OC);

        std::vector<double> rescale(OC, 1.0);
        for (size_t i = 0; i < rescale.size(); i++) {
            rescale[i] = (weightsQuantScales[i] * inQuantScale[i]) / outQuantScale[i];
        }

        const auto ppeConverter = VPU::NCESparsity::ppeConvertersMap.at(arch);
        return [rescale = std::move(rescale), ppeConverter, inElemType, ppe, updateMultForPPE, arch,
                isSpecialRelu](size_t oc) {
            const auto quantScale = rescale[oc];

            const auto scaleApproximation = QuantizationApproximation(arch, quantScale);
            VPUX_THROW_WHEN(isSpecialRelu && (scaleApproximation.shift() < 3),
                            "The curernt PWL solution for leakyRelu with alpha 0.25 requires shift >= 3, but the "
                            "actual shift is {0}",
                            scaleApproximation.shift());
            const auto shift = isSpecialRelu ? scaleApproximation.shift() - 3 : scaleApproximation.shift();

            auto multShift = ppeConverter(checked_cast<uint8_t>(shift),
                                          checked_cast<uint16_t>(scaleApproximation.mult()), rescale[oc], inElemType);
            updateMultForPPE(multShift, ppe);
            return multShift;
        };
    }

    VPUX_THROW("In/Out element type of NCE op mismatch. Both types must be quantized or not quantized. Got: in type "
               "{0}, out type {1}",
               inElemType, outElemType);
}

}  // namespace

int32_t vpux::VPU::NCESparsity::toHex(double realVal) {
    union f32toint32 {
        int32_t m_i32;
        float m_f32;
    };

    // Conversion from double to float leads to precision loss with the order of magnitude 1e-07.
    // It doesn't have any impact on accuracy validation, however checked_cast cannot be used here.
    // See E#49853 for details.
    f32toint32 biasVal;
    biasVal.m_f32 = static_cast<float>(realVal);
    return biasVal.m_i32;
}

const EnumMap<VPU::ArchKind, VPU::NCESparsity::PPEConverterCb> vpux::VPU::NCESparsity::ppeConvertersMap = {
        {VPU::ArchKind::VPUX30XX, getVPUX30XX_Scale},
        {VPU::ArchKind::VPUX311X, getVPUX30XX_Scale},
        {VPU::ArchKind::VPUX37XX, getVPUX37XX_Scale},
};

const EnumMap<VPU::ArchKind, VPU::NCESparsity::BiasConverterCb> vpux::VPU::NCESparsity::biasConvertersMap = {
        {VPU::ArchKind::VPUX30XX, toFixedPoint},
        {VPU::ArchKind::VPUX311X, toFixedPoint},
        {VPU::ArchKind::VPUX37XX, VPU::NCESparsity::toHex},
};

int64_t vpux::VPU::NCESparsity::getBitPatternSize(Mode mode, ShapeRef kernelSize, int64_t SX, mlir::Type elemType,
                                                  int64_t IC) {
    VPUX_THROW_UNLESS(kernelSize.size() == 2, "Unsupported kernel size: %d", kernelSize.size());

    const auto KY = kernelSize[Dims4D::Kernel::Y];
    const auto KX = kernelSize[Dims4D::Kernel::X];

    const auto actualType = getBaseStorageType(elemType);
    const auto windowSize = getWindowSize(KX, SX, actualType);

    VPUX_THROW_UNLESS(windowSize >= KX, "windowsSize must be greater than or equal to KX. windowsSize={0}, KX={1}",
                      windowSize, KX);

    const auto numOfRepeat = mode == VPU::NCESparsity::Mode::CM_CONV ? KY * IC : KY;

    return numOfRepeat * windowSize;
}

int64_t vpux::VPU::NCESparsity::getActivationWindowSize(Mode mode, ShapeRef kernelSize, int64_t SX, mlir::Type elemType,
                                                        int64_t IC) {
    // Align each activation map entry to 128 bits to abide the hw restriction.
    // MaxPool is supported only in depth wise mode.
    // Depth wise does not support weights sparsity in the real sense, but it will have to have an activation window
    // pointer, which is regarded as "fake sparsity".
    size_t perChannelSparsitySize = 0;
    if (mode == Mode::CM_CONV) {
        VPUX_THROW_UNLESS(kernelSize.size() == 2, "Unsupported kernel size: %d", kernelSize.size());
        const auto KX = kernelSize[Dims4D::Kernel::X];

        const auto actualType = getBaseStorageType(elemType);
        const auto windowSize = getWindowSize(KX, SX, actualType);
        const auto windowSparsitySize = std::ceil(windowSize / 8.0);
        const auto numberOfRowsSparsityBytes = std::ceil((KX * IC * windowSparsitySize) / 16.0);

        perChannelSparsitySize = static_cast<size_t>(numberOfRowsSparsityBytes * 16.0);
    } else if (mode == Mode::DW_CONV || mode == Mode::POOL) {
        const auto bitPatternSize = getBitPatternSize(mode, kernelSize, SX, elemType, IC);

        perChannelSparsitySize = static_cast<size_t>(std::ceil(bitPatternSize / 128.0) * 16.0);
    } else {
        VPUX_THROW("Unsupported FakeSparsity mode");
    }

    return perChannelSparsitySize;
}

Shape vpux::VPU::NCESparsity::inferActivationWindowShape(int64_t fakeSparsitySize) {
    return Shape{1, 1, 1, fakeSparsitySize};
}

Shape vpux::VPU::NCESparsity::inferActivationWindowShape(Mode mode, ShapeRef kernelSize, int64_t SX,
                                                         mlir::Type elemType, int64_t IC) {
    const auto activationWindowSize = getActivationWindowSize(mode, kernelSize, SX, elemType, IC);
    return inferActivationWindowShape(activationWindowSize);
}

std::vector<uint8_t> vpux::VPU::NCESparsity::getFakeSparsity(Mode mode, ShapeRef kernelSize, int64_t SX,
                                                             mlir::Type elemType, int64_t IC) {
    const auto actualType = getBaseStorageType(elemType);
    const auto windowSize = getWindowSize(kernelSize[Dims4D::Kernel::X], SX, actualType);
    const auto bitPattern = getBitPattern(mode, kernelSize, windowSize, IC);

    // Align each activation map entry to 128 bits to abide the hw restriction.
    // MaxPool is supported only in depth wise mode.
    // Depth wise does not support weights sparsity in the real sense, but it will have to have an activation window
    // pointer, which is regarded as "fake sparsity".
    size_t perChannelSparsitySize = 0;
    if (mode == Mode::CM_CONV) {
        const auto windowSparsitySize = std::ceil(windowSize / 8.0);
        const auto numberOfRowsSparsityBytes =
                std::ceil((kernelSize[Dims4D::Kernel::X] * IC * windowSparsitySize) / 16.0);
        perChannelSparsitySize = static_cast<size_t>(numberOfRowsSparsityBytes * 16.0);
    } else if (mode == Mode::DW_CONV || mode == Mode::POOL) {
        perChannelSparsitySize = static_cast<size_t>(std::ceil(bitPattern.size() / 128.0) * 16.0);
    } else {
        VPUX_THROW("Unsupported FakeSparsity mode");
    }

    // Repackaging each byte from bitPattern to a bit from fakeSparsity, the rest of the bits remain zero.
    std::vector<uint8_t> perChannelSparsity(perChannelSparsitySize, 0);
    for (auto i : irange(bitPattern.size())) {
        const auto dstInd = (i / 128) * 16 + (i % 128) / 8;
        VPUX_THROW_UNLESS(dstInd < perChannelSparsity.size(),
                          "Attempt to access index '{0}' of perChannelSparsity, which is out of range '{1}'", dstInd,
                          perChannelSparsity.size());
        perChannelSparsity[dstInd] |= bitPattern[i] << (i % 8);
    }

    return perChannelSparsity;
}

int32_t vpux::VPU::NCESparsity::getWeightPtrStep(mlir::Value weights) {
    if (weights == nullptr) {
        return 0;
    }

    const auto filterShape = getShape(weights);

    const auto IC = filterShape[Dims4D::Filter::IC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    // For Channel major and depthwise convolution case, weights table
    // contains both activation window and weights.
    // Check that weights have expected alignment.
    // Other than that, weight step is the same for both z-major (OYXI) and depthwise convolutions.
    const auto origFilterType = weights.getType().cast<vpux::NDTypeInterface>();
    const auto convAlignment = VPU::NCEInvariant::getAlignment(origFilterType.getElementType());
    const auto weightsElementCount = IC * KY * KX;
    VPUX_THROW_UNLESS(
            weightsElementCount % convAlignment == 0,
            "Convolution, Channel Major and Depthwise convolution weights size must be a multiple of {0}, got {1}",
            convAlignment, weightsElementCount);

    const Byte eltSize = getElemTypeSize(weights.getType());
    return checked_cast<int32_t>(IC * KY * KX * eltSize.count());
}

bool isQuantLeakyRelu025(VPU::ArchKind arch, mlir::Type inElemType, mlir::Type outElemType,
                         vpux::IE::PostOp postOpAttr) {
    if (arch == VPU::ArchKind::VPUX30XX || arch == VPU::ArchKind::VPUX311X) {
        return true;
    }

    if (postOpAttr == nullptr) {
        return false;
    }

    bool isQuantized =
            inElemType.isa<mlir::quant::UniformQuantizedType>() && outElemType.isa<mlir::quant::UniformQuantizedType>();
    if (!isQuantized) {
        return false;
    }

    if (postOpAttr.name().getValue() != IE::LeakyReluOp::getOperationName()) {
        return false;
    }

    IE::LeakyReluOp::Adaptor leakyRelu(None, postOpAttr.attrs());
    auto dummyLocation = mlir::UnknownLoc::get(inElemType.getContext());
    if (leakyRelu.verify(dummyLocation).failed()) {
        return false;
    }

    const auto reluSlope = leakyRelu.negative_slope().convertToDouble();
    return isFloatEqual(static_cast<float>(reluSlope), 0.25);
}

std::vector<int32_t> vpux::VPU::NCESparsity::getWeightsTable(mlir::Type inElemType, mlir::Type outElemType,
                                                             Optional<int32_t> weightsPtr, int32_t weightsPtrStep,
                                                             Optional<int32_t> sparsityPtr, int32_t sparsityPtrStep,
                                                             VPU::ArchKind arch, int64_t OC, mlir::Type weightsElemType,
                                                             Const::ContentAttr bias, VPU::PPETaskAttr ppe,
                                                             vpux::IE::PostOp postOpAttr) {
    auto weightsPtrOffset = weightsPtr.hasValue() ? weightsPtr.getValue() : 0;

    // In case of dense operation use sparsityPtrOffset beyond CMX memory range to satisfy HW requirements
    auto sparsityPtrOffset = sparsityPtr.hasValue() ? sparsityPtr.getValue() : SPARSITY_PTR_WHEN_NO_SPARSITY;

    SmallVector<int32_t> weightsPtrs(OC, 0);
    SmallVector<int32_t> sparsityPtrs(OC, 0);
    for (auto oc : irange(OC)) {
        weightsPtrs[oc] = weightsPtrOffset;
        weightsPtrOffset += weightsPtrStep;

        sparsityPtrs[oc] = sparsityPtrOffset;
        sparsityPtrOffset += sparsityPtrStep;
    }

    return getWeightsTable(inElemType, outElemType, weightsPtrs, sparsityPtrs, arch, OC, weightsElemType, bias, ppe,
                           postOpAttr);
}

std::vector<int32_t> vpux::VPU::NCESparsity::getWeightsTable(mlir::Type inElemType, mlir::Type outElemType,
                                                             SmallVector<int32_t> weightsPtrs,
                                                             SmallVector<int32_t> sparsityPtrs, ArchKind arch,
                                                             int64_t OC, mlir::Type weightsElemType,
                                                             Const::ContentAttr bias, VPU::PPETaskAttr ppe,
                                                             vpux::IE::PostOp postOpAttr) {
    VPUX_THROW_WHEN(inElemType == nullptr || outElemType == nullptr,
                    "Can't create weights table without operation input/output types");
    VPUX_THROW_WHEN(static_cast<int64_t>(weightsPtrs.size()) != OC,
                    "Weights pointers size {0} different than output channels {1}", weightsPtrs.size(), OC);
    VPUX_THROW_WHEN(static_cast<int64_t>(sparsityPtrs.size()) != OC,
                    "Sparsity pointers size {0} different than output channels {1}", sparsityPtrs.size(), OC);

    // mixed precision for VPUX30XX
    if (inElemType.isa<mlir::quant::QuantizedType>() && !outElemType.isa<mlir::quant::QuantizedType>() &&
        arch == ArchKind::VPUX30XX) {
        const auto newScale = static_cast<double>(std::pow(2, -16));
        const int64_t zeroPoint = 0;

        auto qType = inElemType.cast<mlir::quant::QuantizedType>();
        outElemType = mlir::quant::UniformQuantizedType::get(qType.getFlags(), qType.getStorageType(),
                                                             qType.getExpressedType(), newScale, zeroPoint,
                                                             qType.getStorageTypeMin(), qType.getStorageTypeMax());
    }

    const auto isSpecialRelu = isQuantLeakyRelu025(arch, inElemType, outElemType, postOpAttr);
    auto getMultShift = getMultShiftFunc(inElemType, outElemType, weightsElemType, ppe, arch, checked_cast<size_t>(OC),
                                         isSpecialRelu);
    auto getBiasFP = getBiasFunc(inElemType, outElemType, weightsElemType, bias, arch, checked_cast<size_t>(OC));

    std::vector<std::int32_t> weightsTableVals(OC * VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC, 0);

    loop_1d(LoopExecPolicy::Parallel, checked_cast<size_t>(OC), [&](const size_t oc) {
        const auto wtInd = oc * static_cast<size_t>(VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC);

        VPUX_THROW_UNLESS(weightsPtrs[oc] % ALIGNMENT_REQUIREMENT_IN_ELEMENTS == 0,
                          "weightsPtrs[{0}] must be multiple of {1}, got {2}", oc, ALIGNMENT_REQUIREMENT_IN_ELEMENTS,
                          weightsPtrs[oc]);
        VPUX_THROW_UNLESS(sparsityPtrs[oc] == SPARSITY_PTR_WHEN_NO_SPARSITY ||
                                  sparsityPtrs[oc] % ALIGNMENT_REQUIREMENT_IN_ELEMENTS == 0,
                          "sparsityPtrs[{0}] must be aligned to {1}, got {2}", oc, ALIGNMENT_REQUIREMENT_IN_ELEMENTS,
                          sparsityPtrs[oc]);

        weightsTableVals[wtInd + 0] = weightsPtrs[oc];
        weightsTableVals[wtInd + 1] = sparsityPtrs[oc];
        weightsTableVals[wtInd + 2] = getMultShift(oc);
        weightsTableVals[wtInd + 3] = getBiasFP(oc);
    });

    return weightsTableVals;
}

std::vector<int32_t> vpux::VPU::NCESparsity::patchWeightsTableSparsityPtrs(
        const std::vector<std::int32_t>& weightsTableVals, const int32_t sparsityPtrOffset,
        const int32_t sparsityPtrStep) {
    const int64_t OC = weightsTableVals.size() / VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC;

    std::vector<std::int32_t> newWeightsTableVals(weightsTableVals.begin(), weightsTableVals.end());

    VPUX_THROW_UNLESS(sparsityPtrOffset % ALIGNMENT_REQUIREMENT_IN_ELEMENTS == 0,
                      "sparsityPtrOffset must be aligned to {0}, got {1}", ALIGNMENT_REQUIREMENT_IN_ELEMENTS,
                      sparsityPtrOffset);

    VPUX_THROW_UNLESS(sparsityPtrStep % ALIGNMENT_REQUIREMENT_IN_ELEMENTS == 0,
                      "sparsityPtrStep must be aligned to {0}, got {1}", ALIGNMENT_REQUIREMENT_IN_ELEMENTS,
                      sparsityPtrStep);

    int32_t offset = sparsityPtrOffset;
    for (auto oc : irange(checked_cast<size_t>(OC))) {
        const auto wtInd = oc * static_cast<size_t>(VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC);

        newWeightsTableVals[wtInd + 1] = offset;

        offset += sparsityPtrStep;
    }

    return newWeightsTableVals;
}

SmallVector<int32_t> vpux::VPU::NCESparsity::getInstructionListTable(ArrayRef<int> rangeAttr, ArrayRef<int> shiftAttr,
                                                                     ArrayRef<int> biasAttr) {
    // NOTE : The instruction list has 5 bits of addresses so the biggest count of instructions is 11111 = 27
    // 27 of course will be aligned to 32 and will contain NOPS inside

    auto range = to_small_vector(rangeAttr);
    auto shift = to_small_vector(shiftAttr);
    auto bias = to_small_vector(biasAttr);

    VPUX_THROW_UNLESS(range.size() == shift.size() + 1 && bias.size() == shift.size(),
                      "One instruction list table is incomplet: range={0} shift={1} bias={2}", range.size(),
                      shift.size(), bias.size());

    range.resize(9, range.back());
    shift.resize(8, shift.back());
    bias.resize(8, shift.back());

    // NOTE: first 2 addresses are hardware reserved areas
    const int32_t ADDR_OF_RESERVED = 6;
    const int32_t VALUE_RESERVED = 0;
    const int32_t ADDR_OF_ADDR_FLEX = 11;
    const int32_t VALUE_ADDR_FLEX = 8;
    // NOTE-END
    const int32_t ADDR_OF_FIRST2_BITS = 9;
    const int32_t ADDR_OF_REST_BITS = 16;
    const int32_t ADDR_OF_VALUE = 19;
    const int32_t MASK_FIRST2_BITS = 3;
    const int32_t ALU_HALT_OPCODE = 6;
    const int32_t ALU_LOAD = 2;
    const int32_t INSTRUCTION_END = 0;
    int32_t first2Bits, first3Bits;
    const int32_t sizeRange = static_cast<int32_t>(range.size());
    const int32_t sizeShift = static_cast<int32_t>(shift.size());
    const int32_t sizeBias = static_cast<int32_t>(bias.size());
    const int32_t fullSize = sizeRange + sizeShift + sizeBias;
    const int32_t noopCount = fullSize >> 4;

    const int32_t size = alignVal<int32_t>(fullSize + noopCount, 16);

    SmallVector<int32_t> templateTable(size - noopCount - 1, ALU_HALT_OPCODE);

    const auto generateTableElement = [&](const int32_t input, const int32_t first2Bits,
                                          const int32_t first3Bits) -> int32_t {
        return ((input << ADDR_OF_VALUE) | (first3Bits << ADDR_OF_REST_BITS) | (VALUE_ADDR_FLEX << ADDR_OF_ADDR_FLEX) |
                (first2Bits << ADDR_OF_FIRST2_BITS) | (VALUE_RESERVED << ADDR_OF_RESERVED) | ALU_LOAD);
    };

    // Populate the instruction list from the table
    // Example:
    //
    // range = {-15, -13, -11, -9, -7, -5, -3, 0, 252}
    // shift = {2, 0, 2, 4, 2, 3, 1, 0}
    // bias = {1, 10, 1, -1, 1, 0, 1, 0}
    //
    // expectedOutput=* = {-7847934, -6798846, -5749758, -4700670, -3588094, -2539006, -1489918, 83458, 132268034,
    // 1196546, 148482, 1197570, 2310146, 1262082, 1786882, 6, 738818, 278530,
    // 803330, 5522434, 804354, -180222, 868866, 345090, 869890, 409602, 0, 6, 6, 6, 6, 6}
    //
    // first 9 values (first line) are for range, next 8 (second line) should be for shift but 9 + 8 > 15,
    // so we need to add ALU_HALT_OPCODE=6 to the 16 position and to continue with the remaining 2 values
    // for shift. Next 8 values (last line) are for bias, followed by INSTRUCTION_END=0, and after this
    // we need to add ALU_HALT_OPCODE=6 until buffer.size()=32.

    for (int32_t j = 0; j < fullSize; j++) {
        first2Bits = j & MASK_FIRST2_BITS;
        first3Bits = j >> 2;
        if (j < sizeRange) {
            templateTable[j] = generateTableElement(range[j], first2Bits, first3Bits);
        } else if (j < sizeRange + sizeShift) {
            templateTable[j] = generateTableElement(shift[j - sizeRange], first2Bits, first3Bits);
        } else {
            templateTable[j] = generateTableElement(bias[j - sizeRange - sizeShift], first2Bits, first3Bits);
        }
    }

    templateTable.insert(templateTable.begin() + fullSize, INSTRUCTION_END);

    if (noopCount > 0) {
        // insert ALU_HALT_OPCODE at the end of the first chain of 16 bytes
        templateTable.insert(templateTable.begin() + 15, ALU_HALT_OPCODE);
    }

    return templateTable;
}

Shape vpux::VPU::NCESparsity::inferWeightsTableShape(int64_t OC) {
    return Shape{OC, 1, 1, VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC};
}

Shape vpux::VPU::NCESparsity::inferWeightsSparsityMapShape(ShapeRef dataShape) {
    VPUX_THROW_UNLESS(dataShape.size() == 4, "Expected data shape to be 4D, while shape is {0}", dataShape);
    const auto workloadSize = std::accumulate(dataShape.begin() + 1, dataShape.end(), static_cast<int64_t>(1),
                                              std::multiplies<int64_t>());
    const auto alignment = Byte(16).to<Bit>().count();
    const auto alignedWorkloadSize = vpux::alignVal(workloadSize, alignment);
    return Shape({dataShape.raw()[0], 1, 1, alignedWorkloadSize});
}

mlir::FailureOr<SmallVector<double>> vpux::VPU::NCESparsity::getRescaledBias(Const::ContentAttr biasAttr,
                                                                             mlir::Type inElemType,
                                                                             mlir::Type filterElemType, int64_t OC) {
    auto inQuantScale = inElemType.isa<mlir::quant::QuantizedType>() ? extractScalesAndZeroPoints(inElemType).first
                                                                     : SmallVector<double>{1.0};
    auto filterQuantScales = filterElemType.isa<mlir::quant::QuantizedType>()
                                     ? extractScalesAndZeroPoints(filterElemType).first
                                     : SmallVector<double>{1.0};
    broadcast(inQuantScale, OC);
    broadcast(filterQuantScales, OC);

    SmallVector<double> rescaledBias(OC, 1.0);
    std::transform(filterQuantScales.begin(), filterQuantScales.end(), inQuantScale.begin(), rescaledBias.begin(),
                   std::multiplies<>());

    auto biasContent = biasAttr.fold();
    auto biasValueRange = biasContent.getValues<double>();
    VPUX_THROW_UNLESS(biasValueRange.size() >= static_cast<size_t>(OC), "bias size {} is less than OC {}",
                      biasValueRange.size(), OC);

    std::transform(biasValueRange.begin(), biasValueRange.begin() + OC, rescaledBias.begin(), rescaledBias.begin(),
                   std::divides<>());

    const auto isValueOutOfRange = llvm::any_of(rescaledBias, [](double newBiasData) {
        return newBiasData <= std::numeric_limits<int32_t>::min() || newBiasData >= std::numeric_limits<int32_t>::max();
    });
    if (isValueOutOfRange) {
        return mlir::failure();
    }
    return rescaledBias;
}

template <typename T>
int64_t countSparseValues(Const::details::ContentRange<T> values, int64_t numChannels, int64_t channelSize,
                          std::function<bool(T)> elemComparisonFn) {
    SmallVector<int64_t> numValsPerOC(numChannels);
    loop_1d(LoopExecPolicy::Parallel, numChannels, [&](size_t oc) {
        const auto channelStart = values.begin() + oc * channelSize;
        for (auto ch = 0; ch < channelSize; ++ch) {
            numValsPerOC[oc] += elemComparisonFn(*(channelStart + ch));
        }
    });
    return std::accumulate(numValsPerOC.begin(), numValsPerOC.end(), 0ll);
}

double vpux::VPU::NCESparsity::getSparsityRatio(Const::DeclareOp weightsConst) {
    const auto content = weightsConst.content();
    const auto contentType = content.getType();
    const auto elemType = contentType.getElementType();

    const auto OC = contentType.getShape()[Dims4D::Filter::OC];
    const auto IC = contentType.getShape()[Dims4D::Filter::IC];
    const auto KY = contentType.getShape()[Dims4D::Filter::KY];
    const auto KX = contentType.getShape()[Dims4D::Filter::KX];
    const auto weightSetSize = IC * KY * KX;

    int64_t numSparseValues = 0;

    if (elemType.isa<mlir::FloatType>()) {
        const auto values = content.getValues<double>();
        numSparseValues = countSparseValues<double>(values, OC, weightSetSize, [](double value) {
            return isDoubleEqual(value, 0.0);
        });
    } else if (elemType.isa<mlir::IntegerType, mlir::quant::QuantizedType>()) {
        const auto values = content.getValues<int64_t>();

        int64_t sparseValue = 0;
        if (const auto qType = elemType.dyn_cast<mlir::quant::UniformQuantizedType>()) {
            sparseValue = qType.getZeroPoint();
        } else if (const auto qType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
            const auto zeroPoints = qType.getZeroPoints();
            if (std::adjacent_find(zeroPoints.begin(), zeroPoints.end(), std::not_equal_to<>()) != zeroPoints.end()) {
                return false;
            }
            sparseValue = zeroPoints[0];
        }

        numSparseValues = countSparseValues<int64_t>(values, OC, weightSetSize, [&sparseValue](int64_t value) {
            return value == sparseValue;
        });
    }

    const auto actualSparsityRatio =
            checked_cast<double>(numSparseValues) / checked_cast<double>(contentType.getNumElements());

    return actualSparsityRatio;
}

bool vpux::VPU::NCESparsity::isSparsifiableWeightsOperand(mlir::Value operand) {
    const auto operandType = operand.getType();
    // already sparse
    if (operandType.isa<VPU::SparseTensorType>()) {
        return false;
    }
    auto sourceOp = operand.getDefiningOp<Const::DeclareOp>();
    if (!sourceOp) {
        return false;
    }
    for (const auto transformation : sourceOp.contentAttr().getTransformations()) {
        if (transformation.isa<Const::SparsifyAttr, Const::GetSparsityMapAttr>()) {
            VPUX_THROW("Trying to sparsify already sparsity related content at '{0}'", sourceOp->getLoc());
        }
    }
    return true;
}
