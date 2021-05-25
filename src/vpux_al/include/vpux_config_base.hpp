//
// Copyright 2020 Intel Corporation.
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

#pragma once

#include <ie_common.h>

#include <vpu/parsed_config_base.hpp>

namespace vpux {

class VPUXConfigBase : public vpu::ParsedConfigBase {
public:
    VPUXConfigBase();
    const std::map<std::string, std::string>& getConfig() const {
        return _config;
    }
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
