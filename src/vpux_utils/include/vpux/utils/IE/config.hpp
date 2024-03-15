//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <utility>
#include "vpux/utils/IE/logger_adapter.hpp"
#include "vpux/utils/core/common_string_utils.hpp"
#include "vpux/utils/core/exceptions.hpp"
#include "vpux/utils/core/type_traits.hpp"

#include <cassert>
#include <chrono>
#include <functional>
#include <iomanip>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace vpux {

//
// OptionParser
//

template <typename T>
struct OptionParser;

template <>
struct OptionParser<std::string> final {
    static std::string parse(std::string_view val) {
        return {val.data(), val.size()};
    }
};

template <>
struct OptionParser<bool> final {
    static bool parse(std::string_view val);
};

template <>
struct OptionParser<int64_t> final {
    static int64_t parse(std::string_view val);
};

template <>
struct OptionParser<uint64_t> final {
    static uint64_t parse(std::string_view val);
};

template <>
struct OptionParser<double> final {
    static double parse(std::string_view val);
};

template <>
struct OptionParser<LogLevel> final {
    static LogLevel parse(std::string_view val);
};

template <typename T>
struct OptionParser<std::vector<T>> final {
    static std::vector<T> parse(std::string_view val) {
        std::vector<T> res;
        splitStringList(val, ',', [&](std::string_view item) {
            res.push_back(OptionParser<T>::parse(item));
        });
        return res;
    }
};

template <typename Rep, typename Period>
struct OptionParser<std::chrono::duration<Rep, Period>> final {
    static std::chrono::duration<Rep, Period> parse(std::string_view val) {
        std::istringstream stream(val.data());

        Rep count{};
        if (stream >> count) {
            CORE_VPUX_THROW_UNLESS(count >= 0, "Value '%ld' is not a valid time duration, non-negative values expected",
                                   count);

            return std::chrono::duration<Rep, Period>(count);
        }

        CORE_VPUX_THROW("Can't parse '%s' as time duration", val.data());
    }
};

//
// OptionPrinter
//

template <typename T>
struct OptionPrinter final {
    static std::string toString(const T& val) {
        std::stringstream ss;
        if constexpr (std::is_floating_point_v<std::decay_t<T>>) {
            ss << std::fixed << std::setprecision(2) << val;
        } else if constexpr (std::is_enum_v<std::decay_t<T>>) {
            ss << stringifyEnum(val);
            return ss.str();
        } else {
            ss << val;
        }
        return ss.str();
    }
};

// NB: boolean config option has values YES for true, NO for false
template <>
struct OptionPrinter<bool> final {
    static std::string toString(bool val);
};

template <typename Rep, typename Period>
struct OptionPrinter<std::chrono::duration<Rep, Period>> final {
    static std::string toString(const std::chrono::duration<Rep, Period>& val) {
        return std::to_string(val.count());
    }
};

template <>
struct OptionPrinter<LogLevel> final {
    static std::string toString(LogLevel val);
};

//
// OptionMode
//

enum class OptionMode {
    Both,
    CompileTime,
    RunTime,
};

std::string_view stringifyEnum(OptionMode val);

//
// OptionBase
//

// Actual Option description must inherit this class and pass itself as template parameter.
template <class ActualOpt, typename T>
struct OptionBase {
    using ValueType = T;

    // `ActualOpt` must implement the following method:
    // static std::string_view key()

    static constexpr std::string_view getTypeName() {
        if constexpr (vpux::TypePrinter<T>::hasName()) {
            return vpux::TypePrinter<T>::name();
        }
        static_assert(vpux::TypePrinter<T>::hasName(),
                      "Options type is not a standard type, please add `getTypeName()` to your option");
    }
    // Overload this to provide environment variable support.
    static std::string_view envVar() {
        return "";
    }

    // Overload this to provide deprecated keys names.
    static std::vector<std::string_view> deprecatedKeys() {
        return {};
    }

    // Overload this to provide default value if it wasn't specified by user.
    // If it is std::nullopt - exception will be thrown in case of missing option access.
    static std::optional<T> defaultValue() {
        return std::nullopt;
    }

    // Overload this to provide more specific parser.
    static ValueType parse(std::string_view val) {
        return OptionParser<ValueType>::parse(val);
    }

    // Overload this to provide more specific validation
    static void validateValue(const ValueType&) {
    }

    // Overload this to provide more specific implementation.
    static OptionMode mode() {
        return OptionMode::Both;
    }

    // Overload this for private options.
    static bool isPublic() {
        return true;
    }

    static std::string toString(const ValueType& val) {
        return OptionPrinter<ValueType>::toString(val);
    }
};

//
// OptionValue
//

namespace details {

class OptionValue {
public:
    virtual ~OptionValue();

    virtual std::string_view getTypeName() const = 0;
    virtual std::string toString() const = 0;
};

template <typename Opt, typename T>
class OptionValueImpl final : public OptionValue {
    using ToStringFunc = std::string (*)(const T&);

public:
    template <typename U>
    OptionValueImpl(U&& val, ToStringFunc toStringImpl): _val(std::forward<U>(val)), _toStringImpl(toStringImpl) {
    }

    std::string_view getTypeName() const final {
        if constexpr (vpux::TypePrinter<T>::hasName()) {
            return vpux::TypePrinter<T>::name();
        } else {
            return Opt::getTypeName();
        }
    }

    const T& getValue() const {
        return _val;
    }

    std::string toString() const override {
        return _toStringImpl(_val);
    }

private:
    T _val;
    ToStringFunc _toStringImpl = nullptr;
};

}  // namespace details

//
// OptionConcept
//

namespace details {

struct OptionConcept final {
    std::string_view (*key)() = nullptr;
    std::string_view (*envVar)() = nullptr;
    OptionMode (*mode)() = nullptr;
    bool (*isPublic)() = nullptr;
    std::shared_ptr<OptionValue> (*validateAndParse)(std::string_view val) = nullptr;
};

template <class Opt>
std::shared_ptr<OptionValue> validateAndParse(std::string_view val) {
    using ValueType = typename Opt::ValueType;

    try {
        auto parsedVal = Opt::parse(val);
        Opt::validateValue(parsedVal);
        return std::make_shared<OptionValueImpl<Opt, ValueType>>(std::move(parsedVal), &Opt::toString);
    } catch (const std::exception& e) {
        CORE_VPUX_THROW("Failed to parse '%s' option : %s", Opt::key().data(), e.what());
    }
}

template <class Opt>
OptionConcept makeOptionModel() {
    return {&Opt::key, &Opt::envVar, &Opt::mode, &Opt::isPublic, &validateAndParse<Opt>};
}

}  // namespace details

//
// OptionsDesc
//

class OptionsDesc final {
public:
    OptionsDesc() = default;
    OptionsDesc(const OptionsDesc&) = default;
    OptionsDesc& operator=(const OptionsDesc&) = default;

    // Destructor preserves unload order of implementation object and reference to library.
    // To preserve destruction order inside default generated assignment operator we store `_impl` before `_so`.
    // And use destructor to remove implementation object before reference to library explicitly.
    ~OptionsDesc() {
        _impl.clear();
    }

public:
    template <class Opt>
    void add();

    void addSharedObject(const std::shared_ptr<void>& so) {
        _so.push_back(so);
    }

public:
    std::vector<std::string> getSupported(bool includePrivate = false) const;

public:
    details::OptionConcept get(std::string_view key, OptionMode mode) const;
    void walk(std::function<void(const details::OptionConcept&)> cb) const;

private:
    std::unordered_map<std::string, details::OptionConcept> _impl;
    std::unordered_map<std::string, std::string> _deprecated;

    // Keep pointer to `_so` to avoid shared library unloading prior destruction of the `_impl` object.
    std::vector<std::shared_ptr<void>> _so;
};

template <class Opt>
void OptionsDesc::add() {
    CORE_VPUX_THROW_UNLESS(_impl.count(Opt::key().data()) == 0, "Option '%s' was already registered",
                           Opt::key().data());
    _impl.insert({Opt::key().data(), details::makeOptionModel<Opt>()});

    for (const auto& deprecatedKey : Opt::deprecatedKeys()) {
        CORE_VPUX_THROW_UNLESS(_deprecated.count(deprecatedKey.data()) == 0, "Option '%s' was already registered",
                               deprecatedKey.data());
        _deprecated.insert({deprecatedKey.data(), Opt::key().data()});
    }
}

//
// Config
//

class Config final {
public:
    using ConfigMap = std::map<std::string, std::string>;
    using ImplMap = std::unordered_map<std::string, std::shared_ptr<details::OptionValue>>;

    explicit Config(const std::shared_ptr<const OptionsDesc>& desc);

    void update(const ConfigMap& options, OptionMode mode = OptionMode::Both);

    void parseEnvVars();

    template <class Opt>
    bool has() const;

    template <class Opt>
    typename Opt::ValueType get() const;

    template <class Opt>
    typename std::string getString() const;

    std::string toString() const;

private:
    std::shared_ptr<const OptionsDesc> _desc;
    ImplMap _impl;
};

template <class Opt>
bool Config::has() const {
    return _impl.count(Opt::key().data()) != 0;
}

template <class Opt>
typename Opt::ValueType Config::get() const {
    using ValueType = typename Opt::ValueType;

    auto log = LoggerAdapter("Config");
    log.trace("Get value for the option '%s'", Opt::key().data());

    const auto it = _impl.find(Opt::key().data());

    if (it == _impl.end()) {
        const std::optional<ValueType> optional = Opt::defaultValue();
        log.trace("The option '%s' was not set by user, try default value", Opt::key().data());

        CORE_VPUX_THROW_UNLESS(optional.has_value(), "Option '%s' was not provided, no default value is available",
                               Opt::key().data());
        return optional.value();
    }

    CORE_VPUX_THROW_WHEN(it->second == nullptr, "Got NULL OptionValue for '%s'", Opt::key().data());

    const auto optVal = std::dynamic_pointer_cast<details::OptionValueImpl<Opt, ValueType>>(it->second);
#if defined(__CHROMIUMOS__)
    if (optVal == nullptr) {
        if (Opt::getTypeName() == it->second->getTypeName()) {
            const auto val = std::static_pointer_cast<details::OptionValueImpl<Opt, ValueType>>(it->second);
            return val->getValue();
        }
    }
#endif
    CORE_VPUX_THROW_WHEN(optVal == nullptr, "Option '%s' has wrong parsed type: expected '%s', got '%s'",
                         Opt::key().data(), Opt::getTypeName().data(), it->second->getTypeName().data());

    return optVal->getValue();
}

template <class Opt>
typename std::string Config::getString() const {
    typename Opt::ValueType value = Config::get<Opt>();

    return Opt::toString(value);
}

//
// envVarStrToBool
//

bool envVarStrToBool(const char* varName, const char* varValue);

}  // namespace vpux
