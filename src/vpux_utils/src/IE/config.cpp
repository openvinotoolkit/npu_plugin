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

#include "vpux/utils/IE/config.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <cpp_interfaces/exception2status.hpp>
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
                          NOT_FOUND_str, key);

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
