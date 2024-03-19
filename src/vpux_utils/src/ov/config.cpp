//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/IE/config.hpp"

#include <openvino/core/except.hpp>

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

using namespace vpux;

//
// OptionParser
//

bool vpux::OptionParser<bool>::parse(std::string_view val) {
    if (val == "YES") {
        return true;
    } else if (val == "NO") {
        return false;
    }

    CORE_VPUX_THROW("Value '%s' is not a valid BOOL option", val.data());
}

int64_t OptionParser<int64_t>::parse(std::string_view val) {
    try {
        return std::stoll(val.data());
    } catch (...) {
        CORE_VPUX_THROW("Value '%s' is not a valid INT64 option", val.data());
    }
}

uint64_t OptionParser<uint64_t>::parse(std::string_view val) {
    try {
        return std::stoull(val.data());
    } catch (...) {
        CORE_VPUX_THROW("Value '%s' is not a valid UINT64 option", val.data());
    }
}

double OptionParser<double>::parse(std::string_view val) {
    try {
        return std::stod(val.data());
    } catch (...) {
        CORE_VPUX_THROW("Value '%s' is not a valid FP64 option", val.data());
    }
}

LogLevel vpux::OptionParser<LogLevel>::parse(std::string_view val) {
    if (val == "LOG_NONE") {
        return LogLevel::None;
    } else if (val == "LOG_ERROR") {
        return LogLevel::Error;
    } else if (val == "LOG_WARNING") {
        return LogLevel::Warning;
    } else if (val == "LOG_INFO") {
        return LogLevel::Info;
    } else if (val == "LOG_DEBUG") {
        return LogLevel::Debug;
    } else if (val == "LOG_TRACE") {
        return LogLevel::Trace;
    }

    CORE_VPUX_THROW("Value '%s' is not a valid LOG_LEVEL option: "
                    "LOG_TRACE, LOG_DEBUG, LOG_INFO, LOG_WARNING, LOG_ERROR, LOG_NONE",
                    val.data());
}

//
// OptionPrinter
//

std::string vpux::OptionPrinter<bool>::toString(bool val) {
    return val ? "YES" : "NO";
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

std::string_view vpux::stringifyEnum(OptionMode val) {
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

vpux::details::OptionConcept vpux::OptionsDesc::get(std::string_view key, OptionMode mode) const {
    auto log = LoggerAdapter("OptionsDesc");

    std::string searchKey{key};
    const auto itDeprecated = _deprecated.find(std::string(key));
    if (itDeprecated != _deprecated.end()) {
        searchKey = itDeprecated->second;
        log.warning("Deprecated option '%s' was used, '%s' should be used instead", key.data(), searchKey.c_str());
    }

    const auto itMain = _impl.find(searchKey);
    CORE_VPUX_THROW_WHEN(itMain == _impl.end(), "[ NOT_FOUND ] Option '%s' is not supported for current configuration",
                         key.data());

    const auto& desc = itMain->second;

    if (mode == OptionMode::RunTime) {
        if (desc.mode() == OptionMode::CompileTime) {
            log.warning("%s option '%s' was used in %s mode", stringifyEnum(desc.mode()).data(), key.data(),
                        stringifyEnum(mode).data());
        }
    }

    return desc;
}

std::vector<std::string> vpux::OptionsDesc::getSupported(bool includePrivate) const {
    std::vector<std::string> res;
    res.reserve(_impl.size());

    for (const auto& p : _impl) {
        if (p.second.isPublic() || includePrivate) {
            res.push_back(p.first);
        }
    }

    return res;
}

void vpux::OptionsDesc::walk(std::function<void(const details::OptionConcept&)> cb) const {
    for (const auto& opt : _impl | map_values) {
        cb(opt);
    }
}

//
// Config
//

vpux::Config::Config(const std::shared_ptr<const OptionsDesc>& desc): _desc(desc) {
    CORE_VPUX_THROW_WHEN(_desc == nullptr, "Got NULL OptionsDesc");
}

void vpux::Config::parseEnvVars() {
    auto log = LoggerAdapter("Config");

    _desc->walk([&](const details::OptionConcept& opt) {
        if (!opt.envVar().empty()) {
            if (const auto envVar = std::getenv(opt.envVar().data())) {
                log.trace("Update option '%s' to value '%s' parsed from environment variable '%s'", opt.key().data(),
                          envVar, opt.envVar().data());

                _impl[opt.key().data()] = opt.validateAndParse(envVar);
            }
        }
    });
}

void vpux::Config::update(const ConfigMap& options, OptionMode mode) {
    auto log = LoggerAdapter("Config");

    for (const auto& p : options) {
        log.trace("Update option '%s' to value '%s'", p.first.c_str(), p.second.c_str());

        const auto opt = _desc->get(p.first, mode);
        _impl[opt.key().data()] = opt.validateAndParse(p.second);
    }
}

std::string vpux::Config::toString() const {
    std::stringstream resultStream;
    for (auto it = _impl.cbegin(); it != _impl.cend(); ++it) {
        const auto key = it->first;

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
        OPENVINO_THROW(std::string("Environment variable ") + varName + " has wrong value : " + e.what());
    }
}
