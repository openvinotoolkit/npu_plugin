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
auto irange(T startIndex, T size) {
    return details::IntegerValuesRange<T>(llvm::None, startIndex, size);
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
