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
#include <vpu/parsed_config_base.hpp>

namespace vpu {
namespace KmbPlugin {

class KmbConfig final : public ParsedConfigBase {
public:
    KmbConfig();

    std::map<std::string, std::string> getParsedConfig() const { return _config; }
    unsigned int numberOfSIPPShaves = 4;

protected:
    const std::unordered_set<std::string>& getCompileOptions() const override;
    const std::unordered_set<std::string>& getRunTimeOptions() const override;
    void parse(const std::map<std::string, std::string>& config) override;

private:
    std::map<std::string, std::string> _config;
};

}  // namespace KmbPlugin
}  // namespace vpu
