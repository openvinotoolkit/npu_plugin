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

#include <mcm_config.h>

#include <map>
#include <string>
#include <unordered_set>

namespace vpu {

class HDDL2Config final : public MCMConfig {
public:
    const std::string& platform() const { return _platform; }
    const std::string& device_id() const { return _device_id; }
    bool performance_counting() const { return _performance_counting; }

    // FIXME [Workaround to change default log lvl]
    LogLevel logLevel() const;

protected:
    const std::unordered_set<std::string>& getCompileOptions() const override;
    const std::unordered_set<std::string>& getRunTimeOptions() const override;
    void parse(const std::map<std::string, std::string>& config) override;

private:
    std::string _platform = "HDDL2";
    LogLevel _logLevel = LogLevel::Warning;
    std::string _device_id;
    bool _performance_counting = false;
};

}  // namespace vpu
