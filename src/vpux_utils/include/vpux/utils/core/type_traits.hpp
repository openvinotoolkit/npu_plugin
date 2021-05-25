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
// Extra helpers for STL type_traits (partially from C++17 and above)
//

#pragma once

#include <type_traits>

namespace vpux {

//
// remove_* & decay
//

template <typename T>
using remove_reference_t = typename std::remove_reference<T>::type;

template <typename T>
using remove_pointer_t = typename std::remove_pointer<T>::type;

template <typename T>
using remove_const_t = typename std::remove_const<T>::type;

template <typename T>
using decay_t = typename std::decay<T>::type;

//
// Bool logic
//

template <bool value>
using bool_c = std::integral_constant<bool, value>;

template <typename T>
using not_ = bool_c<!T::value>;

template <typename...>
struct or_;

template <>
struct or_<> final : std::true_type {};

template <typename T>
struct or_<T> final : T {};

template <typename T0, typename T1>
struct or_<T0, T1> final : bool_c<T0::value || T1::value> {};

template <typename T0, typename... T>
struct or_<T0, T...> final : or_<T0, or_<T...>> {};

template <typename...>
struct and_;

template <>
struct and_<> final : std::true_type {};

template <typename T>
struct and_<T> final : T {};

template <typename T0, typename T1>
struct and_<T0, T1> : bool_c<T0::value && T1::value> {};

template <typename T0, typename... T>
struct and_<T0, T...> final : and_<T0, and_<T...>> {};

//
// enable_if
//

template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

template <typename T, typename... Args>
using enable_t = enable_if_t<and_<Args...>::value, T>;

template <typename... Args>
using require_t = enable_t<void, Args...>;

}  // namespace vpux
