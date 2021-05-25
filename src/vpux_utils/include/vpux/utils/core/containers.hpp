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
// Various helper functions for STL containers.
//

#pragma once

#include <utility>

namespace vpux {

//
// addToContainer - generic way to add element to the container.
//

namespace details {

template <class Container, typename T>
auto addToContainerImpl(Container& cont, T&& val, int) -> decltype(cont.push_back(std::forward<T>(val))) {
    return cont.push_back(std::forward<T>(val));
}

template <class Container, typename T>
auto addToContainerImpl(Container& cont, T&& val, ...) -> decltype(cont.insert(std::forward<T>(val))) {
    return cont.insert(std::forward<T>(val));
}

}  // namespace details

template <class Container, typename T>
void addToContainer(Container& cont, T&& val) {
    details::addToContainerImpl(cont, std::forward<T>(val), 0);
}

}  // namespace vpux
