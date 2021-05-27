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
#include <ie_plugin_config.hpp>

#include <vpu/utils/logger.hpp>

#include <unordered_set>

namespace vpux {

VPU_DECLARE_ENUM(ConfigMode, Any, RunTime)

class VPUXConfigBase {
public:
    vpu::LogLevel logLevel() const {
        return _logLevel;
    }

    bool exclusiveAsyncRequests() const {
        return _exclusiveAsyncRequests;
    }

public:
    VPUXConfigBase() = default;
    virtual ~VPUXConfigBase() = default;

    const std::map<std::string, std::string>& getConfig() const {
        return _config;
    }
    virtual const std::unordered_set<std::string>& getCompileOptions() const;
    virtual const std::unordered_set<std::string>& getRunTimeOptions() const;
    void expandSupportedCompileOptions(const std::unordered_set<std::string>& options);
    void expandSupportedRunTimeOptions(const std::unordered_set<std::string>& options);
    void update(const std::map<std::string, std::string>& config, ConfigMode mode = ConfigMode::Any);

protected:
    virtual void parse(const std::map<std::string, std::string>& config);

protected:
    static std::unordered_set<std::string> merge(const std::unordered_set<std::string>& set1,
                                                 const std::unordered_set<std::string>& set2) {
        auto out = set1;
        out.insert(set2.begin(), set2.end());
        return out;
    }

    static void setOption(std::string& dst, const std::map<std::string, std::string>& config, const std::string& key) {
        const auto value = config.find(key);
        if (value != config.end()) {
            dst = value->second;
        }
    }

    template <typename T, class SupportedMap>
    static void setOption(T& dst, const SupportedMap& supported, const std::map<std::string, std::string>& config,
                          const std::string& key) {
        const auto value = config.find(key);
        if (value != config.end()) {
            const auto parsedValue = supported.find(value->second);
            if (parsedValue == supported.end()) {
                IE_THROW() << "Unsupported value "
                           << "\"" << value->second << "\""
                           << " for key " << key;
            }

            dst = parsedValue->second;
        }
    }

    template <typename T, class PreprocessFunc>
    static void setOption(T& dst, const std::map<std::string, std::string>& config, const std::string& key,
                          const PreprocessFunc& preprocess) {
        const auto value = config.find(key);
        if (value != config.end()) {
            try {
                dst = preprocess(value->second);
            } catch (const std::exception& e) {
                IE_THROW() << "Invalid value "
                           << "\"" << value->second << "\""
                           << " for key " << key << " : " << e.what();
            }
        }
    }

    static int parseInt(const std::string& src) {
        const auto val = std::stoi(src);

        return val;
    }

protected:
    std::unordered_set<std::string> _compileOptions = {CONFIG_KEY(LOG_LEVEL)};
    std::unordered_set<std::string> _runTimeOptions = {CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), CONFIG_KEY(LOG_LEVEL)};

    const std::unordered_map<std::string, bool> switches = {{CONFIG_VALUE(YES), true}, {CONFIG_VALUE(NO), false}};

private:
    std::map<std::string, std::string> _config;
    vpu::Logger::Ptr _log;

private:
    vpu::LogLevel _logLevel = vpu::LogLevel::Error;
    bool _exclusiveAsyncRequests = false;
};

}  // namespace vpux
