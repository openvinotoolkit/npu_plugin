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

#define UNUSED(var) (void)var

#include <map>
#include <string>
#include <unordered_set>

#include <vpu/parsed_config.hpp>
#include <vpu/private_plugin_config.hpp>

namespace vpu {
namespace KmbPlugin {

struct KmbConfig final : ParsedConfig {
    bool forceReset = false;
    int watchdogInterval = 1000;

    explicit KmbConfig(const std::map<std::string, std::string> &config = std::map<std::string, std::string>(),
                          ConfigMode mode = ConfigMode::DEFAULT_MODE);

    std::unordered_set<std::string> getCompileOptions() const override;
    std::map<std::string, std::string> getDefaultConfig() const override;

    const std::map<std::string, std::string>& getParsedConfig() const {
        return _parsedConfig;
    }

private:
    void checkInvalidValues(const std::map<std::string, std::string> &config) const final;
    std::map<std::string, std::string> _parsedConfig;
};

}  // namespace KmbPlugin
}  // namespace vpu
