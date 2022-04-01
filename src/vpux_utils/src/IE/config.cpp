//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/utils/IE/config.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <ie_common.h>
#include <ie_plugin_config.hpp>

using namespace vpux;

//
// OptionParser
//

bool vpux::OptionParser<bool>::parse(StringRef val) {
    if (val == CONFIG_VALUE(YES)) {
        return true;
    } else if (val == CONFIG_VALUE(NO)) {
        return false;
    }

    VPUX_THROW("Value '{0}' is not a valid BOOL option");
}

int64_t OptionParser<int64_t>::parse(StringRef val) {
    try {
        return std::stoll(val.str());
    } catch (...) {
        VPUX_THROW("Value '{0}' is not a valid INT64 option", val);
    }
}

double OptionParser<double>::parse(StringRef val) {
    try {
        return std::stod(val.str());
    } catch (...) {
        VPUX_THROW("Value '{0}' is not a valid FP64 option", val);
    }
}

LogLevel vpux::OptionParser<LogLevel>::parse(StringRef val) {
    if (val == CONFIG_VALUE(LOG_NONE)) {
        return LogLevel::None;
    } else if (val == CONFIG_VALUE(LOG_ERROR)) {
        return LogLevel::Error;
    } else if (val == CONFIG_VALUE(LOG_WARNING)) {
        return LogLevel::Warning;
    } else if (val == CONFIG_VALUE(LOG_INFO)) {
        return LogLevel::Info;
    } else if (val == CONFIG_VALUE(LOG_DEBUG)) {
        return LogLevel::Debug;
    } else if (val == CONFIG_VALUE(LOG_TRACE)) {
        return LogLevel::Trace;
    }

    VPUX_THROW("Value '{0}' is not a valid LOG_LEVEL option", val);
}

//
// OptionPrinter
//

std::string vpux::OptionPrinter<bool>::toString(bool val) {
    return val ? CONFIG_VALUE(YES) : CONFIG_VALUE(NO);
}

std::string vpux::OptionPrinter<LogLevel>::toString(LogLevel val) {
    switch (val) {
    case LogLevel::None:
        return "LOG_NONE";
    case LogLevel::Fatal:
    case LogLevel::Error:
        return "LOG_ERROR";
    case LogLevel::Warning:
        return "LOG_WARNING";
    case LogLevel::Info:
        return "LOG_INFO";
    case LogLevel::Debug:
        return "LOG_DEBUG";
    case LogLevel::Trace:
        return "LOG_TRACE";
    default:
        return "LOG_NONE";
    }
}

//
// OptionMode
//

StringLiteral vpux::stringifyEnum(OptionMode val) {
    switch (val) {
    case OptionMode::Both:
        return "Both";
    case OptionMode::CompileTime:
        return "CompileTime";
    case OptionMode::RunTime:
        return "RunTime";
    default:
        return "<UNKNOWN>";
    }
}

//
// OptionValue
//

vpux::details::OptionValue::~OptionValue() = default;

//
// OptionsDesc
//

vpux::details::OptionConcept vpux::OptionsDesc::get(StringRef key, OptionMode mode) const {
    auto log = Logger::global().nest("OptionsDesc", 0);

    auto searchKey = key;

    const auto itDeprecated = _deprecated.find(key);
    if (itDeprecated != _deprecated.end()) {
        searchKey = itDeprecated->second;
        log.warning("Deprecated option '{0}' was used, '{1}' should be used instead", key, searchKey);
    }

    const auto itMain = _impl.find(searchKey);
    VPUX_THROW_WHEN(itMain == _impl.end(), "{0} Option '{1}' is not supported for current configuration",
                    InferenceEngine::details::ExceptionTraits<InferenceEngine::NotFound>::string(), key);

    const auto& desc = itMain->second;

    if (mode == OptionMode::RunTime) {
        if (desc.mode() == OptionMode::CompileTime) {
            log.warning("{0} option '{1}' was used in {2} mode", desc.mode(), key, mode);
        }
    }

    return desc;
}

std::vector<std::string> vpux::OptionsDesc::getSupported(bool includePrivate) const {
    std::vector<std::string> res;
    res.reserve(_impl.size());

    for (const auto& p : _impl) {
        if (p.second.isPublic() || includePrivate) {
            res.push_back(p.first.str());
        }
    }

    return res;
}

void vpux::OptionsDesc::walk(FuncRef<void(const details::OptionConcept&)> cb) const {
    for (const auto& opt : _impl | map_values) {
        cb(opt);
    }
}

//
// Config
//

vpux::Config::Config(const std::shared_ptr<const OptionsDesc>& desc): _desc(desc) {
    VPUX_THROW_WHEN(_desc == nullptr, "Got NULL OptionsDesc");
}

void vpux::Config::parseEnvVars() {
    auto log = Logger::global().nest("Config", 0);

    _desc->walk([&](const details::OptionConcept& opt) {
        if (!opt.envVar().empty()) {
            if (const auto envVar = std::getenv(opt.envVar().data())) {
                log.trace("Update option '{0}' to value '{1}' parsed from environment variable '{2}'", opt.key(),
                          envVar, opt.envVar());

                _impl[opt.key()] = opt.validateAndParse(envVar);
            }
        }
    });
}

void vpux::Config::update(const ConfigMap& options, OptionMode mode) {
    auto log = Logger::global().nest("Config", 0);

    for (const auto& p : options) {
        log.trace("Update option '{0}' to value '{1}'", p.first, p.second);

        const auto opt = _desc->get(p.first, mode);
        _impl[opt.key()] = opt.validateAndParse(p.second);
    }
}

std::string vpux::Config::toString() const {
    std::stringstream resultStream;
    for (auto it = _impl.cbegin(); it != _impl.cend(); ++it) {
        const auto key = it->first.str();

        resultStream << key << "=\"" << it->second->toString() << "\"";
        if (std::next(it) != _impl.end()) {
            resultStream << " ";
        }
    }

    return resultStream.str();
}

//
// envVarStrToBool
//

bool vpux::envVarStrToBool(const char* varName, const char* varValue) {
    try {
        const auto intVal = std::stoi(varValue);
        if (intVal != 0 && intVal != 1) {
            throw std::invalid_argument("Only 0 and 1 values are supported");
        }
        return (intVal != 0);
    } catch (const std::exception& e) {
        IE_THROW() << "Environment variable " << varName << " has wrong value : " << e.what();
    }
}
