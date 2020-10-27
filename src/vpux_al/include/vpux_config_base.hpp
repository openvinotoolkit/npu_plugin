//
// Copyright 2020 Intel Corporation.
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

#include <vpu/parsed_config_base.hpp>

namespace vpux {

class VPUXConfigBase : public vpu::ParsedConfigBase {
public:
    VPUXConfigBase();
    const std::map<std::string, std::string>& getConfig() const { return _config; }
    const std::unordered_set<std::string>& getCompileOptions() const override;
    const std::unordered_set<std::string>& getRunTimeOptions() const override;
    void expandSupportedCompileOptions(const std::unordered_set<std::string>& options);
    void expandSupportedRunTimeOptions(const std::unordered_set<std::string>& options);

protected:
    void parse(const std::map<std::string, std::string>& config) override;
    std::unordered_set<std::string> _compileOptions;
    std::unordered_set<std::string> _runTimeOptions;

private:
    std::map<std::string, std::string> _config;
};

}  // namespace vpux
