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

#include "vpux_config_base.hpp"

vpux::VPUXConfigBase::VPUXConfigBase()
    : _compileOptions(vpu::ParsedConfigBase::getCompileOptions()),
      _runTimeOptions(vpu::ParsedConfigBase::getRunTimeOptions()) {}

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

const std::unordered_set<std::string>& vpux::VPUXConfigBase::getCompileOptions() const { return _compileOptions; }

const std::unordered_set<std::string>& vpux::VPUXConfigBase::getRunTimeOptions() const { return _runTimeOptions; }
