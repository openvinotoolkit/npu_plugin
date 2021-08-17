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

#include "vpux/compiler/dialect/VPUIP/nce_sparsity.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/enums.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"

using namespace vpux;
using namespace VPUIP;

namespace {

int64_t getWindowSize(int64_t kernelW, int64_t strideW, mlir::Type elemType) {
    VPUX_THROW_UNLESS(kernelW <= 11, "Unsupported kernel size {0}. Supported size up to 11", kernelW);

    // Select the maximum window size not exceeding 32 bytes
    // by iterating through the MPE_NUM values (2, 4, 8, 16)

    VPUX_THROW_UNLESS(elemType.isInteger(8) || elemType.isF16(), "Supported only I8 and FP16 types");

    // Only MPE0, MPE4, MPE8 and MPE12 support FP16 data format
    const int mpeNumLimit = elemType.isF16() ? 4 : 16;

    const Bit typeSizeInBits = getElemTypeSize(elemType);

    // Window size is limited to 32 bytes by HW. Size of the data type
    // needs to be accounted to find the max (32 for U8, 16 for FP16)
    const int64_t maxWindowSize = 32 / (typeSizeInBits.count() / 8);

    int64_t windowSize = 0;
    int mpeNum = 2;

    while (mpeNum <= mpeNumLimit) {
        if (strideW <= kernelW) {
            windowSize = kernelW + strideW * (mpeNum - 1);
        } else {
            windowSize = kernelW * mpeNum;
        }

        if (windowSize > maxWindowSize)
            return windowSize;

        mpeNum *= 2;
    }

    return windowSize;
}

std::vector<uint8_t> getBitPattern(mlir::ArrayRef<int64_t> kernelSize, int64_t windowSize) {
    const auto kernelW = kernelSize[0];
    const auto kernelH = kernelSize[1];

    VPUX_THROW_UNLESS(windowSize >= kernelW,
                      "windowsSize must be greater than or equal to kernelW. windowsSize={0}, kernelW={1}", windowSize,
                      kernelW);

    const auto numBitsSet = kernelW;
    const auto numBitsClear = windowSize - kernelW;

    SmallVector<uint8_t> window;
    window.reserve(windowSize);
    window.insert(window.end(), numBitsSet, 1);
    window.insert(window.end(), numBitsClear, 0);

    const auto numOfRepeat = kernelH;

    std::vector<uint8_t> bitPattern;
    bitPattern.reserve(numOfRepeat * windowSize);
    for (auto i = 0; i < numOfRepeat; i++) {
        bitPattern.insert(bitPattern.end(), window.begin(), window.end());
    }

    return bitPattern;
}

constexpr std::int32_t MTL_SPARSITY = 0xFFFFFF;

std::int32_t toFixedPoint(double realVal) {
    // FIXME: 2 ^ 16 might be more obvious
    return std::lround(realVal * 65536.);
}

std::int32_t toHex(double realVal) {
    union f32toint32 {
        std::int32_t m_i32;
        float m_f32;
    };

    f32toint32 biasVal;
    biasVal.m_f32 = static_cast<float>(realVal);
    return biasVal.m_i32;
}

constexpr std::int32_t getKMBScale(double scale = 1.0) {
    (void) scale;

    constexpr std::int32_t PRELU_SCALE_OFFSET = 0;
    constexpr std::int32_t PRELU_SCALE_VALUE = 1;

    // FIXME: PPE shift is actually 6 bit long, 2 higher bits stand for rounding mode
    constexpr std::int32_t PPE_SHIFT_OFFSET = 8;
    constexpr std::int32_t PPE_SHIFT_VALUE = 0;

    constexpr std::int32_t PPE_MULT_OFFSET = 16;
    // FIXME: PPE multiplier has sign, which may affect lower bits
    constexpr std::int32_t PPE_MULT_VALUE = 1;

    constexpr std::int32_t KMB_SCALE = (PRELU_SCALE_VALUE << PRELU_SCALE_OFFSET) | (PPE_SHIFT_VALUE << PPE_SHIFT_OFFSET) |
                                       (PPE_MULT_VALUE << PPE_MULT_OFFSET);

    return KMB_SCALE;
}

void computeQuantMultShift(double scale, std::uint32_t& shift, std::uint32_t& mult) {
    auto bits = 15;
    auto exponent = 0;
    double mantissa = std::frexp(scale, &exponent);
    shift = bits - exponent;
    mult = static_cast<std::uint32_t>((mantissa * pow(2, bits)));
}

std::int32_t getMTLScale(double scale) {
    std::int32_t multshift = 0;
    // 8bit mult mask
    static constexpr std::uint32_t PRELU_MULT_MASK = 0x000000FF;
    // 6bit shift mask
    static constexpr std::uint32_t PRELU_SHIFT_MASK = 0x00003F00;
    static constexpr std::uint32_t PRELU_SHIFT_SHIFT = 8;
    // round mode mask
    static constexpr std::uint32_t ROUND_MODE_MASK = 0x0000C000;
    static constexpr std::uint32_t ROUND_MODE_SHIFT = 14;
    // scale mask
    static constexpr std::uint32_t SCALE_MODE_MASK = 0xFFFF0000;
    static constexpr std::uint32_t SCALE_MODE_SHIFT = 16;

    // harcoded
    std::int32_t round32 = 1;
    std::int32_t reluMult = 0;
    std::uint32_t mult = 0;
    std::uint32_t shift = 0;
    computeQuantMultShift(scale, shift, mult);
    multshift = static_cast<std::int32_t>(
            ((mult << SCALE_MODE_SHIFT) & SCALE_MODE_MASK) | ((round32 << ROUND_MODE_SHIFT) & ROUND_MODE_MASK) |
            ((shift << PRELU_SHIFT_SHIFT) & PRELU_SHIFT_MASK) | (reluMult & PRELU_MULT_MASK));

    return multshift;
}

}  // namespace

const vpux::EnumMap<vpux::VPUIP::ArchKind, vpux::VPUIP::NCESparsity::PPEConverterCb> vpux::VPUIP::NCESparsity::ppeConvertersMap = {
    {vpux::VPUIP::ArchKind::KMB, getKMBScale},
    {vpux::VPUIP::ArchKind::TBH, getKMBScale},
    {vpux::VPUIP::ArchKind::MTL, getMTLScale},
};

const vpux::EnumMap<vpux::VPUIP::ArchKind, vpux::VPUIP::NCESparsity::BiasConverterCb> vpux::VPUIP::NCESparsity::biasConvertersMap = {
    {vpux::VPUIP::ArchKind::KMB, toFixedPoint},
    {vpux::VPUIP::ArchKind::TBH, toFixedPoint},
    {vpux::VPUIP::ArchKind::MTL, toHex},
};

int64_t vpux::VPUIP::NCESparsity::getBitPatternSize(mlir::ArrayRef<int64_t> kernelSize, int64_t strideW,
                                                    mlir::Type elemType) {
    VPUX_THROW_UNLESS(kernelSize.size() == 2, "Unsupported kernel size: %d", kernelSize.size());

    const auto windowSize = getWindowSize(kernelSize[0], strideW, elemType);
    return kernelSize[1] * windowSize;
}

int64_t vpux::VPUIP::NCESparsity::getActivationWindowSize(mlir::ArrayRef<int64_t> kernelSize, int64_t strideW,
                                                          mlir::Type elemType, int64_t inputChannels) {
    const auto bitPatternSize = getBitPatternSize(kernelSize, strideW, elemType);
    const auto perChannelSparsitySize = static_cast<std::size_t>(std::ceil(bitPatternSize / 128.0) * 16);
    const auto activationWindowSize = inputChannels * perChannelSparsitySize;

    return activationWindowSize;
}

std::vector<uint8_t> vpux::VPUIP::NCESparsity::getFakeSparsity(mlir::ArrayRef<int64_t> kernelSize, int64_t strideW,
                                                               mlir::Type elemType, int64_t inputChannels) {
    const auto windowSize = getWindowSize(kernelSize[0], strideW, elemType);
    const auto bitPattern = getBitPattern(kernelSize, windowSize);

    // To align each activation map entry to 16 bytes to abide the hw restriction
    const auto perChannelSparsitySize = static_cast<std::size_t>(std::ceil(bitPattern.size() / 128.0) * 16);

    // MaxPool is supported only in depth wise mode.
    // Depth wise does not support weights sparsity in the real sense,
    // but it will have to have an activation window pointer,
    // which is regarded as "fake sparsity"
    SmallVector<uint8_t> perChannelSparsity;
    perChannelSparsity.resize(perChannelSparsitySize);

    // Repackaging each byte from bitPattern to a bit from fakeSparsity
    // The rest of the bits remain zero
    for (size_t i = 0; i < bitPattern.size(); i++) {
        perChannelSparsity[(i / 128) * 16 + (i % 128) / 8] |= bitPattern[i] << (i % 8);
    }

    std::vector<uint8_t> fakeSparsity;
    fakeSparsity.reserve(inputChannels * perChannelSparsitySize);
    for (auto i = 0; i < inputChannels; i++) {
        fakeSparsity.insert(fakeSparsity.end(), perChannelSparsity.begin(), perChannelSparsity.end());
    }

    return fakeSparsity;
}

std::vector<std::int32_t> vpux::VPUIP::NCESparsity::getWeightsTable(std::int64_t OC, vpux::VPUIP::NCESparsity::GetBiasCb getBiasFP, std::int32_t weightPtrOffset, std::int32_t weightPtrStep,
                                                                    std::int32_t sparsityPtrOffset, vpux::VPUIP::ArchKind arch,
                                                                    mlir::Type inputType, mlir::Type weightsType, mlir::Type outputType) {
    const auto getMultShift = [inputType, weightsType, outputType](vpux::VPUIP::ArchKind architecture) -> std::int32_t {
        const auto ppeConverter = ppeConvertersMap.at(architecture);
        const auto getScale = [](mlir::Type type) -> double {
            if (auto quantized = type.dyn_cast_or_null<mlir::quant::UniformQuantizedType>()) {
                return quantized.getScale();
            } else {
                return 1.0;
            }
        };

        if (architecture == vpux::VPUIP::ArchKind::MTL) {
            const auto inputScale   = getScale(inputType);
            const auto weightsScale = getScale(weightsType);
            const auto outputScale  = getScale(outputType);

            const auto scale = (inputScale * weightsScale) / outputScale;
            return inputType.isBF16() || inputType.isF16() ? toHex(scale) : ppeConverter(scale);
        } else {
            return ppeConverter(1.0);
        }
    };

    const auto multShift = getMultShift(arch);

    const std::int32_t sparsityPtr = arch == vpux::VPUIP::ArchKind::MTL ? MTL_SPARSITY : sparsityPtrOffset;

    const auto convertBias = [&](std::int64_t oc) -> std::int32_t {
        const auto biasVal = getBiasFP(oc);
        const auto biasConverter = biasConvertersMap.at(arch);
        return biasConverter(biasVal);
    };

    std::vector<std::int32_t> weightsTableVals(OC * vpux::VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC, 0);

    for (auto oc : irange(checked_cast<std::size_t>(OC))) {
        const auto wtInd = oc * static_cast<std::size_t>(vpux::VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC);

        weightsTableVals[wtInd + 0] = weightPtrOffset;
        weightsTableVals[wtInd + 1] = sparsityPtr;
        weightsTableVals[wtInd + 2] = multShift;
        weightsTableVals[wtInd + 3] = convertBias(oc);

        weightPtrOffset += weightPtrStep;
    }

    return weightsTableVals;
}
