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
void splitStringList(StringRef str,
                     char delim,
                     FuncRef<void(StringRef)> callback);

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

std::string eraseSubString(StringRef str,
                           StringRef strToRemove,
                           bool removeAllAfterSubstr = false);

}  // namespace vpux
