//
// Copyright Intel Corporation.
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

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/hash.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/string_ref.hpp"
#include "vpux/utils/core/string_utils.hpp"

#include <ie_parameter.hpp>

#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace vpux {

//
// OptionParser
//

template <typename T>
struct OptionParser;

template <>
struct OptionParser<StringRef> final {
    static StringRef parse(StringRef val) {
        return val;
    }
};

template <>
struct OptionParser<bool> final {
    static bool parse(StringRef val);
};

template <>
struct OptionParser<int64_t> final {
    static int64_t parse(StringRef val);
};

template <>
struct OptionParser<double> final {
    static double parse(StringRef val);
};

template <>
struct OptionParser<LogLevel> final {
    static LogLevel parse(StringRef val);
};

template <typename T>
struct OptionParser<std::vector<T>> final {
    static std::vector<T> parse(StringRef val) {
        std::vector<T> res;
        splitStringList(val, ',', [&](StringRef item) {
            res.push_back(OptionParser<T>::parse(item));
        });
        return res;
    }
};

//
// OptionMode
//

enum class OptionMode {
    Both,
    CompileTime,
    RunTime,
};

StringLiteral stringifyEnum(OptionMode val);

//
// OptionBase
//

// Actual Option description must inherit this class and pass itself as template parameter.
template <class ActualOpt, typename T>
struct OptionBase {
    using ValueType = T;

    // `ActualOpt` must implement the following method:
    // static StringRef key()

#ifdef VPUX_DEVELOPER_BUILD
    // Overload this to provide environment variable support.
    static StringRef envVar() {
        return {};
    }
#endif

    // Overload this to provide deprecated keys names.
    static SmallVector<StringRef> deprecatedKeys() {
        return {};
    }

    // Overload this to provide default value if it wasn't specified by user.
    // If it is None - exception will be thrown in case of missing option access.
    static llvm::NoneType defaultValue() {
        return None;
    }

    // Overload this to provide more specific parser.
    static ValueType parse(StringRef val) {
        return OptionParser<ValueType>::parse(val);
    }

    // Overload this to provide more specific validation
    static void validateValue(const ValueType&) {
    }

    static void validate(StringRef val) {
        try {
            const auto parsedVal = ActualOpt::parse(val);
            ActualOpt::validateValue(parsedVal);
        } catch (const std::exception& e) {
            VPUX_THROW("Failed to parse '{0}' option : {1}", ActualOpt::key(), e.what());
        }
    }

    // Overload this to provide more specific implementation.
    static OptionMode mode() {
        return OptionMode::Both;
    }

    // Overload this for private options.
    static bool isPublic() {
        return true;
    }
};

//
// OptionConcept
//

namespace details {

class OptionConcept {
public:
    using Ptr = std::shared_ptr<OptionConcept>;

public:
    virtual ~OptionConcept() = default;

    virtual StringRef key() const = 0;
    virtual void validate(StringRef val) const = 0;
    virtual OptionMode mode() const = 0;
    virtual bool isPublic() const = 0;
};

template <class Opt>
class OptionModel final : public OptionConcept {
public:
    StringRef key() const final {
        return Opt::key();
    }

    void validate(StringRef val) const final {
        Opt::validate(val);
    }

    OptionMode mode() const final {
        return Opt::mode();
    }

    bool isPublic() const final {
        return Opt::isPublic();
    }
};

}  // namespace details

//
// OptionsDesc
//

class OptionsDesc final {
public:
    template <class Opt>
    void addOption();

    const details::OptionConcept& validate(StringRef key, StringRef val, OptionMode mode) const;

    std::vector<std::string> getSupported(bool includePrivate = false) const;

private:
    std::unordered_map<StringRef, details::OptionConcept::Ptr> _impl;
    std::unordered_map<StringRef, details::OptionConcept::Ptr> _deprecated;
};

template <class Opt>
void OptionsDesc::addOption() {
    VPUX_THROW_UNLESS(_impl.count(Opt::key()) == 0, "Option '{0}' was already registered", Opt::key());
    const auto res = _impl.insert({Opt::key(), std::make_shared<details::OptionModel<Opt>>()});

    for (const auto& deprecatedKey : Opt::deprecatedKeys()) {
        VPUX_THROW_UNLESS(_deprecated.count(deprecatedKey) == 0, "Option '{0}' was already registered", deprecatedKey);
        _deprecated.insert({deprecatedKey, res.first->second});
    }
}

//
// Config
//

class Config final {
public:
    using ConfigMap = std::map<std::string, std::string>;
    using ImplMap = std::unordered_map<StringRef, std::string>;

public:
    Config();

public:
    const OptionsDesc& desc() const {
        return *_desc;
    }
    OptionsDesc& desc() {
        return *_desc;
    }

public:
    void update(const ConfigMap& options, OptionMode mode = OptionMode::Both);

public:
    template <class Opt>
    bool has() const;

    template <class Opt>
    typename Opt::ValueType get() const;

private:
    std::shared_ptr<OptionsDesc> _desc;
    ImplMap _impl;
};

template <class Opt>
bool Config::has() const {
#ifdef VPUX_DEVELOPER_BUILD
    if (!Opt::envVar().empty()) {
        if (std::getenv(Opt::envVar().data()) != nullptr) {
            return true;
        }
    }
#endif

    return _impl.count(Opt::key()) != 0;
}

template <class Opt>
typename Opt::ValueType Config::get() const {
#ifdef VPUX_DEVELOPER_BUILD
    if (!Opt::envVar().empty()) {
        if (const auto envVar = std::getenv(Opt::envVar().data())) {
            return Opt::parse(envVar);
        }
    }
#endif

    const auto it = _impl.find(Opt::key());

    if (it == _impl.end()) {
        const Optional<typename Opt::ValueType> res = Opt::defaultValue();
        VPUX_THROW_UNLESS(res.hasValue(), "Option '{0}' was not provided, no default value is available", Opt::key());
        return res.getValue();
    }

    return Opt::parse(it->second);
}

}  // namespace vpux
