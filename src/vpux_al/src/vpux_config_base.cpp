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

#include "vpux_config_base.hpp"

vpux::VPUXConfigBase::VPUXConfigBase()
        : _compileOptions(vpu::ParsedConfigBase::getCompileOptions()),
          _runTimeOptions(vpu::ParsedConfigBase::getRunTimeOptions()) {
}

void vpux::VPUXConfigBase::parse(const std::map<std::string, std::string>& config) {
    ParsedConfigBase::parse(config);
    for (const auto& p : config) {
        auto it = _config.find(p.first);
        // FIXME: insert_or_assign (c++17)
        if (it != _config.end()) {
            it->second = p.second;
        } else {
            _config.insert(p);
        }
    }
}

void vpux::VPUXConfigBase::expandSupportedCompileOptions(const std::unordered_set<std::string>& options) {
    _compileOptions.insert(options.cbegin(), options.cend());
}

void vpux::VPUXConfigBase::expandSupportedRunTimeOptions(const std::unordered_set<std::string>& options) {
    _runTimeOptions.insert(options.cbegin(), options.cend());
}

const std::unordered_set<std::string>& vpux::VPUXConfigBase::getCompileOptions() const {
    return _compileOptions;
}

const std::unordered_set<std::string>& vpux::VPUXConfigBase::getRunTimeOptions() const {
    return _runTimeOptions;
}
