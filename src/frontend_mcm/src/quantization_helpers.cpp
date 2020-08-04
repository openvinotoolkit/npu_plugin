//
// Copyright 2019-2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "quantization_helpers.hpp"

#include <precision_utils.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "ie_utils.hpp"

#ifdef ENABLE_MCM_COMPILER
#include "include/mcm/tensor/quantization_params.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace vpu {

namespace QuantizationHelpers {

int64_t calculateZeroPoint(float high, float low, int levels, InferenceEngine::Precision precision) {
    int64_t zeroPoint = 0;

    // Typical condition for symmetric case is low < 0, high > 0
    if (precision == InferenceEngine::Precision::I8) {
        if ((low <= 0.f) && (high >= 0.f)) {
            float x = -(levels - 1) * ((high + low) * 0.5f) / (high - low);
            zeroPoint = round(x);
        } else if (low > 0.f) {
            zeroPoint = 127 - (levels - 1);  // TODO Why not assert?
        } else if (high < 0.f) {
            zeroPoint = 127;  // TODO Why not assert?
        }
    } else if (precision == InferenceEngine::Precision::U8) {
        //  MCM team provide this formula, need check
        if ((low <= 0.f) && (high >= 0.f)) {
            float x = -(levels - 1) * low / (high - low);
            zeroPoint = round(x);
        } else if (low >= 0.f) {
            zeroPoint = 0;  // TODO Why not assert?
        } else if (high <= 0.f) {
            zeroPoint = (levels - 1);  // TODO Why not assert?
        }
    }

    return zeroPoint;
}

bool isCNNNetworkQuantized(const InferenceEngine::CNNNetwork& network) {
    details::CNNNetworkIterator i(&static_cast<const InferenceEngine::ICNNNetwork&>(network)), end;
    for (; i != end; ++i) {
        auto layer = *i;
        if (layer->type == "FakeQuantize") {
            return true;
        }
    }

    return false;
}

}  // namespace QuantizationHelpers
}  // namespace vpu

#endif
