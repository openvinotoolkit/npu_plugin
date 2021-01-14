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

//
// String format utilities.
//

#pragma once

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/mask.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/slice.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/core/string_ref.hpp"
#include "vpux/utils/core/type_traits.hpp"

#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_os_ostream.h>

#include <array>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <system_error>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cstddef>

namespace vpux {

//
// printTo
//

template <typename... Args>
auto& printTo(llvm::raw_ostream& stream, StringRef format, Args&&... args) {
    return stream << llvm::formatv(format.data(), std::forward<Args>(args)...);
}

template <typename... Args>
auto& printTo(std::ostream& stream, StringRef format, Args&&... args) {
    llvm::raw_os_ostream(stream) << llvm::formatv(format.data(), std::forward<Args>(args)...);
    return stream;
}

template <class Stream, typename... Args>
auto&& printTo(Stream&& stream, StringRef format, Args&&... args) {
    return stream << llvm::formatv(format.data(), std::forward<Args>(args)...).str();
}

//
// Various base classes for `llvm::format_provider` implementations.
//

struct ContainerFormatter {
    template <class Container>
    static void format(const Container& cont, llvm::raw_ostream& stream, StringRef style) {
        stream << '[';

        using IterT = typename Container::const_iterator;
        using IterRange = llvm::iterator_range<IterT>;

        const auto range = make_range(cont);
        llvm::format_provider<IterRange>::format(range, stream, style);

        stream << ']';
    }
};

struct MapFormatter {
    template <class Map>
    static void format(const Map& map, llvm::raw_ostream& stream, StringRef style) {
        stream << '<';

        ptrdiff_t ind = 0;
        const ptrdiff_t size = llvm::size(map);
        for (const auto& p : map) {
            auto keyAdapter = llvm::detail::build_format_adapter(p.first);
            keyAdapter.format(stream, style);

            stream << " : ";

            auto valAdapter = llvm::detail::build_format_adapter(p.second);
            valAdapter.format(stream, style);

            if (ind + 1 < size) {
                stream << ", ";
            }

            ++ind;
        }

        stream << '>';
    }
};

//
// `printFormat` method handling.
//
// It allows to implement `printFormat` method in class without creating
// separate `llvm::format_provider` instance.
//

namespace details {

template <class Obj, typename = void>
struct HasPrintFormat {
    static constexpr bool value = false;
};

template <class Obj>
struct HasPrintFormat<Obj, decltype(std::declval<Obj>().printFormat(std::declval<llvm::raw_ostream&>()))> {
    static constexpr bool value = true;
};

}  // namespace details

//
// `stringifyEnum` function handling.
//

namespace details {

template <class Enum, typename = void>
struct HasStringifyEnum {
    static constexpr bool value = false;
};

template <class Enum>
struct HasStringifyEnum<Enum, require_t<std::is_base_of<StringRef, decltype(stringifyEnum(std::declval<Enum>()))>>> {
    static constexpr bool value = true;
};

}  // namespace details

}  // namespace vpux

//
// Format providers
//

namespace llvm {

template <>
struct format_provider<std::error_code> final {
    static void format(const std::error_code& err, llvm::raw_ostream& stream, StringRef style) {
        auto adapter = llvm::detail::build_format_adapter(err.message());
        adapter.format(stream, style);
    }
};

template <typename T1, typename T2>
struct format_provider<std::pair<T1, T2>> final {
    static void format(const std::pair<T1, T2>& val, llvm::raw_ostream& stream, StringRef style) {
        stream << '(';

        auto adapter1 = llvm::detail::build_format_adapter(val.first);
        adapter1.format(stream, style);

        stream << ", ";

        auto adapter2 = llvm::detail::build_format_adapter(val.second);
        adapter2.format(stream, style);

        stream << ')';
    }
};

template <typename T, size_t Count>
struct format_provider<std::array<T, Count>> final : vpux::ContainerFormatter {};

template <typename T, class A>
struct format_provider<std::vector<T, A>> final : vpux::ContainerFormatter {};

template <typename T, class A>
struct format_provider<std::list<T, A>> final : vpux::ContainerFormatter {};

template <typename T, class C, class A>
struct format_provider<std::set<T, C, A>> final : vpux::ContainerFormatter {};

template <typename T, class H, class P, class A>
struct format_provider<std::unordered_set<T, H, P, A>> final : vpux::ContainerFormatter {};

template <typename K, typename V, class C, class A>
struct format_provider<std::map<K, V, C, A>> final : vpux::MapFormatter {};

template <typename K, typename V, class H, class P, class A>
struct format_provider<std::unordered_map<K, V, H, P, A>> final : vpux::MapFormatter {};

template <typename T>
struct format_provider<ArrayRef<T>> final : vpux::ContainerFormatter {};

template <typename T>
struct format_provider<MutableArrayRef<T>> final : vpux::ContainerFormatter {};

template <typename T>
struct format_provider<SmallVectorImpl<T>> final : vpux::ContainerFormatter {};

template <typename T, unsigned N>
struct format_provider<SmallVector<T, N>> final : vpux::ContainerFormatter {};

template <class Range>
struct format_provider<detail::enumerator<Range>> final : vpux::ContainerFormatter {};

template <typename T>
struct format_provider<vpux::details::IntegerValuesRange<T>> final : vpux::ContainerFormatter {};

template <typename Enum>
struct format_provider<Enum, vpux::require_t<std::is_enum<Enum>, vpux::details::HasStringifyEnum<Enum>>> final {
    static void format(const Enum& val, llvm::raw_ostream& stream, StringRef style) {
        auto adapter = llvm::detail::build_format_adapter(stringifyEnum(val));
        adapter.format(stream, style);
    }
};

template <>
struct format_provider<vpux::Mask> final {
    static void format(const vpux::Mask& mask, llvm::raw_ostream& stream, StringRef style) {
        auto adapter = llvm::detail::build_format_adapter(mask.asRange());
        adapter.format(stream, style);
    }
};

template <>
struct format_provider<vpux::Slice> final {
    static void format(const vpux::Slice& slice, llvm::raw_ostream& stream, StringRef style) {
        stream << '[';

        auto adapter1 = llvm::detail::build_format_adapter(slice.begin());
        adapter1.format(stream, style);

        stream << ", ";

        auto adapter2 = llvm::detail::build_format_adapter(slice.end());
        adapter2.format(stream, style);

        stream << ')';
    }
};

template <typename T>
struct format_provider<Optional<T>> final {
    static void format(const Optional<T>& opt, llvm::raw_ostream& stream, StringRef style) {
        if (opt.hasValue()) {
            auto adapter = llvm::detail::build_format_adapter(opt.getValue());
            adapter.format(stream, style);
        } else {
            stream << "<NONE>";
        }
    }
};

template <class Obj>
struct format_provider<Obj, vpux::require_t<vpux::details::HasPrintFormat<Obj>>> final {
    static void format(const Obj& obj, llvm::raw_ostream& stream, StringRef) {
        obj.printFormat(stream);
    }
};

template <typename... Args>
struct format_provider<std::tuple<Args...>> final {
    static void format(const std::tuple<Args...>& val, llvm::raw_ostream& stream, StringRef style) {
        stream << '<';
        printItems(val, stream, style);
        stream << '>';
    }

private:
    template <size_t Index = 0>
    static auto printItems(const std::tuple<Args...>& val, llvm::raw_ostream& stream, StringRef style)
            -> vpux::enable_if_t<Index + 1 == sizeof...(Args)> {
        auto adapter = llvm::detail::build_format_adapter(std::get<Index>(val));
        adapter.format(stream, style);
    }

    template <size_t Index = 0>
    static auto printItems(const std::tuple<Args...>& val, llvm::raw_ostream& stream, StringRef style)
            -> vpux::enable_if_t<Index + 1 < sizeof...(Args)> {
        auto adapter = llvm::detail::build_format_adapter(std::get<Index>(val));
        adapter.format(stream, style);

        stream << ", ";

        printItems<Index + 1>(val, stream, style);
    }
};

}  // namespace llvm
