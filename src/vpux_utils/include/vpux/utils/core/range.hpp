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

//
// Range concept from C++20.
//

#pragma once

#include "vpux/utils/core/containers.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/core/type_traits.hpp"

#include <llvm/ADT/None.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/iterator_range.h>

#include <vector>

namespace vpux {

//
// IteratorRange - range wrapper for STL iterators and containers.
//

using llvm::make_range;

template <class Range>
auto make_range(Range&& range) {
    return make_range(std::begin(std::forward<Range>(range)), std::end(std::forward<Range>(range)));
}

namespace details {

struct AsRangeTag final {};

template <class Range>
auto operator|(Range&& range, AsRangeTag) {
    return make_range(std::forward<Range>(range));
}

}  // namespace details

constexpr details::AsRangeTag as_range;

//
// Drop first N elements from the Range.
//

using llvm::drop_begin;

//
// Transformed range.
//

using llvm::map_range;

namespace details {

template <class FuncTy>
struct MapRangeTag final {
    FuncTy func;
};

template <class Range, class FuncTy>
auto operator|(Range&& range, MapRangeTag<FuncTy>&& tag) {
    return map_range(std::forward<Range>(range), std::move(tag.func));
}

}  // namespace details

template <class FuncTy>
auto transformed(FuncTy&& func) {
    return details::MapRangeTag<std::remove_reference_t<FuncTy>>{std::forward<FuncTy>(func)};
}

//
// Reversed range.
//

using llvm::reverse;

namespace details {

struct ReverseTag final {};

template <class Range>
auto operator|(Range&& range, ReverseTag) {
    return reverse(std::forward<Range>(range));
}

}  // namespace details

constexpr details::ReverseTag reversed;

//
// Filtered range.
//

using llvm::make_filter_range;

namespace details {

template <class PredicateT>
struct FilterRangeTag final {
    PredicateT pred;
};

template <class Range, class PredicateT>
auto operator|(Range&& range, FilterRangeTag<PredicateT>&& tag) {
    return make_filter_range(std::forward<Range>(range), std::move(tag.pred));
}

}  // namespace details

template <class PredicateT>
auto filtered(PredicateT&& pred) {
    return details::FilterRangeTag<std::remove_reference_t<PredicateT>>{std::forward<PredicateT>(pred)};
}

//
// Zipped range.
//

using llvm::zip;

//
// Concatenated range.
//

using llvm::concat;

//
// Ranges for map keys/values.
//

using llvm::make_first_range;
using llvm::make_second_range;

namespace details {

struct MapKeysTag final {};
struct MapValuesTag final {};

template <class Range>
auto operator|(Range&& range, MapKeysTag) {
    return make_first_range(std::forward<Range>(range));
}

template <class Range>
auto operator|(Range&& range, MapValuesTag) {
    return make_second_range(std::forward<Range>(range));
}

}  // namespace details

constexpr details::MapKeysTag map_keys;
constexpr details::MapValuesTag map_values;

//
// Indexed range.
//

using llvm::enumerate;

namespace details {

struct EnumerateTag final {};

template <class Range>
auto operator|(Range&& range, EnumerateTag) {
    return enumerate(std::forward<Range>(range));
}

}  // namespace details

constexpr details::EnumerateTag indexed;

//
// Integer values range
//

namespace details {

template <class T, typename = require_t<std::is_integral<T>>>
class IntegerValuesRange final : public llvm::indexed_accessor_range<IntegerValuesRange<T>, llvm::NoneType, T, T*, T> {
public:
    using llvm::indexed_accessor_range<IntegerValuesRange<T>, llvm::NoneType, T, T*, T>::indexed_accessor_range;

public:
    static T dereference(const llvm::NoneType&, ptrdiff_t index) {
        return static_cast<T>(index);
    }
};

}  // namespace details

template <class T, typename = require_t<std::is_integral<T>>>
auto irange(T startIndex, T endIndex) {
    assert(endIndex >= startIndex);
    return details::IntegerValuesRange<T>(llvm::None, startIndex, endIndex - startIndex);
}

template <class T, typename = require_t<std::is_integral<T>>>
auto irange(T size) {
    return irange(static_cast<T>(0), size);
}

//
// Copy range to container.
//

using llvm::to_vector;

template <class Range>
auto to_small_vector(Range&& range) {
    using ValueType = std::decay_t<decltype(*std::begin(range))>;
    return SmallVector<ValueType>(std::begin(range), std::end(range));
}

template <class Range>
auto to_std_vector(Range&& range) {
    using ValueType = std::decay_t<decltype(*std::begin(range))>;
    return std::vector<ValueType>(std::begin(range), std::end(range));
}

template <class Container, class Range>
auto to_container(Range&& range) {
    Container out;
    for (auto&& val : range) {
        addToContainer(out, std::forward<decltype(val)>(val));
    }
    return out;
}

}  // namespace vpux
