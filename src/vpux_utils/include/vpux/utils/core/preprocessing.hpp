//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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
