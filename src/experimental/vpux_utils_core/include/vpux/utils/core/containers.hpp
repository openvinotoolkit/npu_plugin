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
auto addToContainerImpl(Container& cont, T&& val, int)
        -> decltype(cont.push_back(std::forward<T>(val))) {
    return cont.push_back(std::forward<T>(val));
}

template <class Container, typename T>
auto addToContainerImpl(Container& cont, T&& val, ...)
        -> decltype(cont.insert(std::forward<T>(val))) {
    return cont.insert(std::forward<T>(val));
}

}  // namespace details

template <class Container, typename T>
void addToContainer(Container& cont, T&& val) {
    details::addToContainerImpl(cont, std::forward<T>(val), 0);
}

}  // namespace vpux
