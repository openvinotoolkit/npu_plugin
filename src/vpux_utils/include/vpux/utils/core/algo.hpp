//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/small_vector.hpp"

namespace vpux {

template <class T>
void broadcast(SmallVectorImpl<T>& values, size_t size) {
    VPUX_THROW_UNLESS(values.size() <= size, "Cannot broadcast to size {0}", size);

    if (values.size() == size) {
        return;
    }

    VPUX_THROW_UNLESS(values.size() == 1, "Broadcast from scalar is only supported");
    values.resize(size, values.front());
}

}  // namespace vpux
