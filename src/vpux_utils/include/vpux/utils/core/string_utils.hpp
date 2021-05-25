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

// Remove all `ch` character occurences in `str` string.
std::string removeCharFromString(StringRef str, char ch);

//
// eraseSubString
//

std::string eraseSubString(StringRef str, StringRef strToRemove, bool removeAllAfterSubstr = false);

}  // namespace vpux
