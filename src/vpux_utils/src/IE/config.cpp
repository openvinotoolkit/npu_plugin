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

#include "vpux/utils/IE/config.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <ie_plugin_config.hpp>

#include <llvm/ADT/StringSwitch.h>

using namespace vpux;
using namespace InferenceEngine;

//
// OptionParser
//

bool vpux::OptionParser<bool>::parse(StringRef val) {
    const auto res = llvm::StringSwitch<Optional<bool>>(val).Case("YES", true).Case("NO", false).Default(None);
    VPUX_THROW_UNLESS(res.hasValue(), "Value '{0}' is not a valid BOOL option");
    return res.getValue();
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
    const auto res = llvm::StringSwitch<Optional<LogLevel>>(val)
                             .Case("LOG_NONE", LogLevel::None)
                             .Case("LOG_ERROR", LogLevel::Error)
                             .Case("LOG_WARNING", LogLevel::Warning)
                             .Case("LOG_INFO", LogLevel::Info)
                             .Case("LOG_DEBUG", LogLevel::Debug)
                             .Case("LOG_TRACE", LogLevel::Trace)
                             .Default(None);
    VPUX_THROW_UNLESS(res.hasValue(), "Value '{0}' is not a valid LOG_LEVEL option");
    return res.getValue();
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
// OptionsDesc
//

const vpux::details::OptionConcept& vpux::OptionsDesc::validate(StringRef key, StringRef val, OptionMode mode) const {
    auto log = Logger::global();

    auto it = _impl.find(key);
    if (it == _impl.end()) {
        it = _deprecated.find(key);
        VPUX_THROW_UNLESS(it != _deprecated.end(), "{0}Option '{1}' is not supported for current configuration",
                          InferenceEngine::details::ExceptionTraits<InferenceEngine::NotFound>::string(), key);

        log.warning("Deprecated option '{0}' was used, '{1}' should be used instead", key, it->second->key());
    }

    const auto& desc = it->second;

    if (mode == OptionMode::RunTime) {
        if (desc->mode() == OptionMode::CompileTime) {
            log.warning("{0} option '{1}' was used in {2} mode", desc->mode(), key, mode);
        }
    }

    desc->validate(val);

    return *desc;
}

std::vector<std::string> vpux::OptionsDesc::getSupported(bool includePrivate) const {
    std::vector<std::string> res;
    res.reserve(_impl.size());

    for (const auto& p : _impl) {
        if (p.second->isPublic() || includePrivate) {
            res.push_back(p.first.str());
        }
    }

    return res;
}

//
// Config
//

vpux::Config::Config() {
    _desc = std::make_shared<OptionsDesc>();
}

void vpux::Config::update(const ConfigMap& options, OptionMode mode) {
    for (const auto& p : options) {
        const auto& opt = _desc->validate(p.first, p.second, mode);
        _impl[opt.key()] = p.second;
    }
}
