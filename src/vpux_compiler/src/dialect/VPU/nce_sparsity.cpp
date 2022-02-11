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

#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/enums.hpp"

using namespace vpux;

namespace {

bool isMixedPrecisionSupported(VPU::ArchKind arch) {
    return arch == VPU::ArchKind::MTL;
}

template <class T>
void broadcast(SmallVectorImpl<T>& values, size_t size) {
    VPUX_THROW_UNLESS(values.size() <= size, "Cannot broadcast to size {0}", size);

    if (values.size() == size) {
        return;
    }

    VPUX_THROW_UNLESS(values.size() == 1, "Broadcast from scalar is only supported");
    values.resize(size, values.front());
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

int32_t toHex(double realVal) {
    union f32toint32 {
        int32_t m_i32;
        float m_f32;
    };

    f32toint32 biasVal;
    biasVal.m_f32 = static_cast<float>(realVal);
    return biasVal.m_i32;
}

constexpr int32_t getKMBScale(unsigned shift, unsigned mult, double, mlir::Type) {
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

int32_t getMTLScale(unsigned shift, unsigned mult, double rescale, mlir::Type inputType) {
    // MTL expects scale in IEEE754 format in NCE_DPU_PPE_FP_SCALE register in case input has FP16/BF16 type
    if (inputType.isF16() || inputType.isBF16() || inputType.isF32()) {
        return toHex(rescale);
    }

    int32_t PRELU_SCALE_OFFSET = 0;
    int32_t PRELU_SCALE_VALUE = 0;

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

llvm::unique_function<int32_t(size_t)> getBiasFunc(mlir::Type inElemType, mlir::Type outElemType,
                                                   mlir::Type weightsElemType, Const::ContentAttr bias,
                                                   VPU::ArchKind arch, size_t OC) {
    if (bias == nullptr) {
        return [](int64_t) -> double {
            return 0.0f;
        };
    }

    auto biasContent = bias.fold();

    if (inElemType.isa<mlir::quant::QuantizedType>() && outElemType.isa<mlir::quant::QuantizedType>()) {
        auto inQuantScale = extractScalesAndZeroPoints(inElemType).first;
        auto weightsQuantScales = exractWeightsScales(weightsElemType);

        broadcast(inQuantScale, OC);
        broadcast(weightsQuantScales, OC);

        std::vector<double> rescale(OC, 1.0);
        std::transform(weightsQuantScales.begin(), weightsQuantScales.end(), inQuantScale.begin(), rescale.begin(),
                       std::multiplies<>());

        return [biasContent = std::move(biasContent), rescale = std::move(rescale)](size_t oc) -> int32_t {
            const auto newBiasData =
                    checked_cast<int64_t>(std::round(biasContent.getValues<double>()[oc] / rescale[oc]));
            VPUX_THROW_UNLESS(newBiasData > std::numeric_limits<int32_t>::min() &&
                                      newBiasData < std::numeric_limits<int32_t>::max(),
                              "Bias value is out of range {0}", newBiasData);

            return static_cast<int32_t>(newBiasData);
        };
    } else if (!inElemType.isa<mlir::quant::QuantizedType>() && !outElemType.isa<mlir::quant::QuantizedType>()) {
        return [biasContent = std::move(biasContent), arch](int64_t oc) -> int32_t {
            const auto biasVal = biasContent.getValues<float>()[oc];
            const auto biasConverter = VPU::NCESparsity::biasConvertersMap.at(arch);
            return biasConverter(biasVal);
        };
    }

    VPUX_THROW("In/Out element type of NCE op mismatch. Both types must be quantized or not quantized. Got: in type "
               "{0}, out type {1}",
               inElemType, outElemType);
}

llvm::unique_function<int32_t(size_t)> getMultShiftFunc(mlir::Type inElemType, mlir::Type outElemType,
                                                        mlir::Type weightsType, VPU::PPETaskAttr ppe,
                                                        VPU::ArchKind arch, size_t OC) {
    auto updateMultForPPE = [](int32_t& mult, VPU::PPETaskAttr ppe) {
        if (ppe && ppe.mode().getValue() == VPU::PPEMode::LPRELU) {
            mult &= 0xFFFFFF00;
            mult |= static_cast<int32_t>(ppe.lrelu_mult().getInt());
        }
    };
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
        return [rescale = std::move(rescale), ppeConverter, inElemType, ppe, updateMultForPPE](size_t oc) {
            uint32_t shift = 0;
            uint32_t mult = 0;
            vpux::VPU::NCESparsity::computeQuantMultShift(rescale[oc], shift, mult);
            int32_t multShift = ppeConverter(shift, mult, rescale[oc], inElemType);
            updateMultForPPE(multShift, ppe);
            return multShift;
        };
    }

    VPUX_THROW("In/Out element type of NCE op mismatch. Both types must be quantized or not quantized. Got: in type "
               "{0}, out type {1}",
               inElemType, outElemType);
}

}  // namespace

const EnumMap<VPU::ArchKind, VPU::NCESparsity::PPEConverterCb> vpux::VPU::NCESparsity::ppeConvertersMap = {
        {VPU::ArchKind::KMB, getKMBScale},
        {VPU::ArchKind::TBH, getKMBScale},
        {VPU::ArchKind::MTL, getMTLScale},
};

const EnumMap<VPU::ArchKind, VPU::NCESparsity::BiasConverterCb> vpux::VPU::NCESparsity::biasConvertersMap = {
        {VPU::ArchKind::KMB, toFixedPoint},
        {VPU::ArchKind::TBH, toFixedPoint},
        {VPU::ArchKind::MTL, toHex},
};

void vpux::VPU::NCESparsity::computeQuantMultShift(double scale, uint32_t& shift, uint32_t& mult, uint32_t bits) {
    int32_t exponent = 0;

    const double mantissa = std::frexp(scale, &exponent);
    shift = bits - exponent;
    mult = static_cast<uint32_t>((mantissa * pow(2, bits)));
}

int64_t vpux::VPU::NCESparsity::getBitPatternSize(Mode mode, ShapeRef kernelSize, int64_t SX, mlir::Type elemType,
                                                  int64_t IC) {
    VPUX_THROW_UNLESS(kernelSize.size() == 2, "Unsupported kernel size: %d", kernelSize.size());

    const auto actualType = getBaseStorageType(elemType);
    const auto windowSize = getWindowSize(kernelSize[Dims4D::Kernel::X], SX, actualType);

    if (mode == Mode::CM_CONV) {
        return kernelSize[Dims4D::Kernel::Y] * windowSize * IC;
    } else if (mode == Mode::DW_CONV || mode == Mode::POOL) {
        return kernelSize[Dims4D::Kernel::Y] * windowSize;
    } else {
        VPUX_THROW("Unsupported FakeSparsity mode");
    }
}

int64_t vpux::VPU::NCESparsity::getActivationWindowSize(Mode mode, ShapeRef kernelSize, int64_t SX, mlir::Type elemType,
                                                        int64_t IC) {
    const auto actualType = getBaseStorageType(elemType);
    const auto bitPatternSize = getBitPatternSize(mode, kernelSize, SX, actualType, IC);
    const auto perChannelSparsitySize = static_cast<size_t>(std::ceil(bitPatternSize / 128.0) * 16);
    const auto activationWindowSize = IC * perChannelSparsitySize;
    return activationWindowSize;
}

std::vector<uint8_t> vpux::VPU::NCESparsity::getFakeSparsity(Mode mode, ShapeRef kernelSize, int64_t SX,
                                                             mlir::Type elemType, int64_t IC, int64_t OC) {
    const auto actualType = getBaseStorageType(elemType);
    const auto windowSize = getWindowSize(kernelSize[Dims4D::Kernel::X], SX, actualType);
    const auto bitPattern = getBitPattern(mode, kernelSize, windowSize, IC);

    // Align each activation map entry to 16 bytes to abide the hw restriction.
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
    SmallVector<uint8_t> perChannelSparsity(perChannelSparsitySize, 0);
    for (auto i : irange(bitPattern.size())) {
        const auto dstInd = (i / 128) * 16 + (i % 128) / 8;
        VPUX_THROW_UNLESS(dstInd < perChannelSparsity.size(),
                          "Attempt to access index '{0}' of perChannelSparsity, which is out of range '{1}'", dstInd,
                          perChannelSparsity.size());
        perChannelSparsity[dstInd] |= bitPattern[i] << (i % 8);
    }

    std::vector<uint8_t> fakeSparsity;
    fakeSparsity.reserve(OC * perChannelSparsitySize);
    for (auto i : irange(OC)) {
        std::ignore = i;
        fakeSparsity.insert(fakeSparsity.end(), perChannelSparsity.begin(), perChannelSparsity.end());
    }

    return fakeSparsity;
}

std::vector<int32_t> vpux::VPU::NCESparsity::getWeightsTable(mlir::Type inElemType, mlir::Type outElemType,
                                                             Optional<int32_t> weightPtr, int32_t weightPtrStep,
                                                             Optional<int32_t> sparsityPtr, VPU::ArchKind arch,
                                                             int64_t OC, mlir::Type weightsElemType,
                                                             Const::ContentAttr bias, VPU::PPETaskAttr ppe) {
    VPUX_THROW_WHEN(inElemType == nullptr || outElemType == nullptr,
                    "Can't create weights table without operation input/output types");

    auto getMultShift = getMultShiftFunc(inElemType, outElemType, weightsElemType, ppe, arch, checked_cast<size_t>(OC));
    auto getBiasFP = getBiasFunc(inElemType, outElemType, weightsElemType, bias, arch, checked_cast<size_t>(OC));

    auto weightPtrOffset = weightPtr.hasValue() ? weightPtr.getValue() : 0;
    std::vector<std::int32_t> weightsTableVals(OC * VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC, 0);

    // In case of dense operation use sparsityPtrOffset beyond CMX memory range to satisfy HW requirements
    auto sparsityPtrOffset = sparsityPtr.hasValue() ? sparsityPtr.getValue() : SPARSITY_PTR_WHEN_NO_SPARISTY;
    int32_t sparsityPtrStep = 0;
    if (arch == VPU::ArchKind::MTL) {
        if (weightsElemType) {
            auto elementBitSize = static_cast<Bit>(getElemTypeSize(weightsElemType));
            sparsityPtrStep = static_cast<int32_t>(
                    static_cast<Byte>(Bit(weightPtrStep * CHAR_BIT / elementBitSize.count())).count());
        } else {
            sparsityPtrOffset = SPARSITY_PTR_WHEN_NO_SPARISTY;
        }
    }

    const auto weightsElementTypeBitSize =
            weightsElemType ? static_cast<Bit>(getElemTypeSize(weightsElemType)).count() : 0;

    for (auto oc : irange(checked_cast<size_t>(OC))) {
        const auto wtInd = oc * static_cast<size_t>(VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC);

        if (weightsElemType) {
            const auto alignment = (ALIGNMENT_REQUIREMENT_IN_ELEMENTS * weightsElementTypeBitSize) / CHAR_BIT;
            VPUX_THROW_UNLESS(weightPtrOffset % alignment == 0,
                              "weightsPtrOffset must be multiple of {0}, got {1} on oc {2}", alignment, weightPtr, oc);
        }

        weightsTableVals[wtInd + 0] = weightPtrOffset;
        weightsTableVals[wtInd + 1] = sparsityPtrOffset;
        weightsTableVals[wtInd + 2] = getMultShift(oc);
        weightsTableVals[wtInd + 3] = getBiasFP(oc);

        weightPtrOffset += weightPtrStep;
        sparsityPtrOffset += sparsityPtrStep;
    }
    return weightsTableVals;
}
