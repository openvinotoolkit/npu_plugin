//
// Copyright (C) 2018-2019 Intel Corporation.
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

#pragma once

#include <string>
#include <map>
#include <vector>

#include <ie_common.h>

#include <vpu/utils/enums.hpp>

namespace vpu {

namespace ie = InferenceEngine;

struct StageMetaInfo final {
    ie::InferenceEngineProfileInfo::LayerStatus status = ie::InferenceEngineProfileInfo::LayerStatus::NOT_RUN;

    std::string layerName;
    std::string layerType;

    std::string stageName;
    std::string stageType;
};

VPU_DECLARE_ENUM(PerfReport,
    PerLayer,
    PerStage
)

std::map<std::string, ie::InferenceEngineProfileInfo> parsePerformanceReport(
        const std::vector<StageMetaInfo>& stagesMeta,
        const float* deviceTimings,
        int deviceTimingsCount,
        PerfReport perfReport,
        bool printReceiveTensorTime);

}  // namespace vpu
