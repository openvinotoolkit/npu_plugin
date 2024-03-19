//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
// Various string manipulation utility functions.
//

#pragma once

#include "vpux/utils/core/containers.hpp"

#include <string.h>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace vpux {

//
// splitStringList
//

// Splits the `str` string onto separate elements using `delim` as delimiter and
// call `callback` for each element.
void splitStringList(const std::string& str, char delim, std::function<void(std::string_view)> callback);

// Splits the string into container.
template <class Container>
Container splitStringList(const std::string& str, char delim) {
    Container out;

    splitStringList(str, delim, [&out](std::string_view val) {
        addToContainer(out, val);
    });

    return out;
}

//
// removeCharFromString
//

// Remove all `ch` character occurrences in `str` string.
std::string removeCharFromString(std::string&& str, char ch);

//
// eraseSubString
//

std::string eraseSubString(std::string&& str, const std::string& strToRemove, bool removeAllAfterSubstr = false);

std::string printFormattedCStr(const char* fmt, ...)
#if defined(__clang__)
        ;
#elif defined(__GNUC__) || defined(__GNUG__)
        __attribute__((format(printf, 1, 2)));
#else
        ;
#endif
}  // namespace vpux
