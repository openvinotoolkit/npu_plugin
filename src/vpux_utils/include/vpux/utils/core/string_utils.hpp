//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

//
// Various string manipulation utility functions.
//

#pragma once

#include "vpux/utils/core/containers.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <string>

namespace vpux {

//
// splitStringList
//

// Splits the `str` string onto separate elements using `delim` as delimiter and
// call `callback` for each element.
void splitStringList(StringRef str, char delim, FuncRef<void(StringRef)> callback);

// Splits the string into container.
template <class Container>
Container splitStringList(StringRef str, char delim) {
    Container out;

    splitStringList(str, delim, [&out](StringRef val) {
        addToContainer(out, val);
    });

    return out;
}

//
// removeCharFromString
//

// Remove all `ch` character occurrences in `str` string.
std::string removeCharFromString(StringRef str, char ch);

//
// eraseSubString
//

std::string eraseSubString(StringRef str, StringRef strToRemove, bool removeAllAfterSubstr = false);

}  // namespace vpux
