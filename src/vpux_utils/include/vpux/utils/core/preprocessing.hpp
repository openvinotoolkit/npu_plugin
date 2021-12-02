//
// Copyright 2021 Intel Corporation.
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

#pragma once

namespace vpux {

enum class PreProcessColorSpace : uint32_t { NONE, BGR, RGB, NV12, I420 };
enum class PreProcessResizeAlgorithm : uint32_t { NO_RESIZE, RESIZE_BILINEAR, RESIZE_AREA };

/**
 * @brief A helper structure describing preprocess info for inputs
 */
struct PreProcessInfo {
    explicit PreProcessInfo(std::string inputName = "", PreProcessColorSpace inputFormat = PreProcessColorSpace::NONE,
                            PreProcessColorSpace outputFormat = PreProcessColorSpace::NONE,
                            PreProcessResizeAlgorithm algorithm = PreProcessResizeAlgorithm::NO_RESIZE)
            : _inputName(std::move(inputName)),
              _inputFormat(inputFormat),
              _outputFormat(outputFormat),
              _algorithm(algorithm){};
    const std::string _inputName;
    const PreProcessColorSpace _inputFormat;
    const PreProcessColorSpace _outputFormat;
    const PreProcessResizeAlgorithm _algorithm;
};

}  // namespace vpux
