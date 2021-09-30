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

//
// Quantization parameters
//

#pragma once

#include <string>
#include <unordered_map>

namespace vpux {

/**
 * @brief Quantization parameters
 */
struct QuantizationParam {
    QuantizationParam(const bool pluginQuantization = false, const float scale = 1.f, const uint8_t zeroPoint = 0)
            : _pluginQuantization(pluginQuantization), _scale(scale), _zeroPoint(zeroPoint) {
    }
    bool _pluginQuantization;
    float _scale;
    uint8_t _zeroPoint;
};

/**
 * @brief Quantization parameters map
 */
using QuantizationParamMap = std::unordered_map<std::string, QuantizationParam>;

}  // namespace vpux
