// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <legacy/ie_layers.h>

namespace vpu {
namespace details {

IE_SUPPRESS_DEPRECATED_START

namespace ie = InferenceEngine;

/**
 * @brief Quantization layer details and basic operations on them.
 */
class QuantizationDetails {
public:
    QuantizationDetails();
    QuantizationDetails(const QuantizationDetails& quantizationDetails);
    QuantizationDetails(const size_t levels, const std::vector<float>& inputLowValues,
                        const std::vector<float>& inputHighValues, const std::vector<float>& outputLowValues,
                        const std::vector<float>& outputHighValues, const size_t inputIntervalsCount,
                        const size_t outputIntervalsCount, const size_t outputChannelsCount);

    static bool outputLayoutIsSupported(const ie::CNNLayer& quantize);

    static void getInputIntervals(const ie::CNNLayer& quantize, std::vector<float>& inputLowValues,
                                  std::vector<float>& inputHighValues, size_t& inputIntervalsCount);

    static void getOutputIntervals(const ie::CNNLayer& quantize, std::vector<float>& outputLowValues,
                                   std::vector<float>& outputHighValues, size_t& outputIntervalsCount);

    static QuantizationDetails getDetails(const ie::CNNLayer& quantize);
    static bool isSupportedLevel(const size_t level);

    const size_t levels;
    const std::vector<float> inputLowValues;
    const std::vector<float> inputHighValues;
    const std::vector<float> outputLowValues;
    const std::vector<float> outputHighValues;
    const size_t inputIntervalsCount;
    const size_t outputIntervalsCount;
    const size_t outputChannelsCount;

private:
    QuantizationDetails& operator=(const QuantizationDetails& /*target*/) {
        return *this;
    }
    static void validate(const ie::CNNLayerPtr& constantLayer);
    static std::vector<float> getBlobValue(const ie::CNNLayerPtr& constantLayer);
};

IE_SUPPRESS_DEPRECATED_END

}  // namespace details
}  // namespace vpu
