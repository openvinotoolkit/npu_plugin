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

void vpux::VPUXConfigBase::parse(const std::map<std::string, std::string>& config) {
    static const std::unordered_map<std::string, vpu::LogLevel> logLevels = {
            {CONFIG_VALUE(LOG_NONE), vpu::LogLevel::None},       {CONFIG_VALUE(LOG_ERROR), vpu::LogLevel::Error},
            {CONFIG_VALUE(LOG_WARNING), vpu::LogLevel::Warning}, {CONFIG_VALUE(LOG_INFO), vpu::LogLevel::Info},
            {CONFIG_VALUE(LOG_DEBUG), vpu::LogLevel::Debug},     {CONFIG_VALUE(LOG_TRACE), vpu::LogLevel::Trace}};

    setOption(_logLevel, logLevels, config, CONFIG_KEY(LOG_LEVEL));
    setOption(_exclusiveAsyncRequests, switches, config, CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS));

#ifndef NDEBUG
    if (const auto envVar = std::getenv("IE_VPUX_LOG_LEVEL")) {
        _logLevel = logLevels.at(envVar);
    }
#endif
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
    static const std::unordered_set<std::string> options = {
            CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS),
            CONFIG_KEY(LOG_LEVEL),
    };

    return _runTimeOptions;
}

void vpux::VPUXConfigBase::update(const std::map<std::string, std::string>& config, ConfigMode mode) {
    const auto& compileOptions = getCompileOptions();
    const auto& runTimeOptions = getRunTimeOptions();

    for (const auto& entry : config) {
        const bool isCompileOption = compileOptions.count(entry.first) != 0;
        const bool isRunTimeOption = runTimeOptions.count(entry.first) != 0;

        if (!isCompileOption && !isRunTimeOption) {
            IE_THROW(NotFound) << entry.first << " key is not supported for VPU";
        }

        if (mode == ConfigMode::RunTime) {
            if (!isRunTimeOption) {
                _log->warning("%s option is used in %s mode", entry.first, mode);
            }
        }
    }

    parse(config);
}
