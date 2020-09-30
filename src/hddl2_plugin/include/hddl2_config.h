//
// Copyright 2019 Intel Corporation.
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

#include <ie_common.h>

#include <hddl2/hddl2_plugin_config.hpp>
#include <map>
#include <string>
#include <unordered_set>
#include <vpux_config.hpp>

namespace vpu {

class HDDL2Config final : public vpux::VPUXConfig {
public:
    InferenceEngine::ColorFormat getGraphColorFormat() const { return _graph_color_format; }
    uint64_t getCSRAMSize() const { return _csram_size; }

    void parse(const std::map<std::string, std::string>& config) override;

protected:
    const std::unordered_set<std::string>& getCompileOptions() const override;
    const std::unordered_set<std::string>& getRunTimeOptions() const override;

private:
    InferenceEngine::ColorFormat _graph_color_format = InferenceEngine::ColorFormat::BGR;
    uint64_t _csram_size = 0;
};

}  // namespace vpu
