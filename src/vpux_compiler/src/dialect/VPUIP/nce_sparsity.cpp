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

}  // namespace

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
