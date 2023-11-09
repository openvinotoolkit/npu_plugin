//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/core/string_utils.hpp"

#include <algorithm>

using namespace vpux;

//
// splitStringList
//

void vpux::splitStringList(StringRef str, char delim, FuncRef<void(StringRef)> callback) {
    const auto begin = str.begin();
    const auto end = str.end();

    auto curBegin = begin;
    auto curEnd = begin;
    while (curEnd != end) {
        while (curEnd != end && *curEnd != delim) {
            ++curEnd;
        }

        callback(StringRef(curBegin, static_cast<size_t>(curEnd - curBegin)));

        if (curEnd != end) {
            ++curEnd;
            curBegin = curEnd;
        }
    }
}

//
// removeCharFromString
//

std::string vpux::removeCharFromString(StringRef str, char ch) {
    auto out = str.str();
    out.erase(std::remove(out.begin(), out.end(), ch), out.end());
    return out;
}

//
// eraseSubString
//

std::string vpux::eraseSubString(StringRef str, StringRef strToRemove, bool removeAllAfterSubstr) {
    auto out = str.str();

    const auto pos = str.find(strToRemove);

    if (pos != StringRef::npos) {
        if (removeAllAfterSubstr) {
            out.erase(pos);
        } else {
            out.erase(pos, strToRemove.size());
        }
    }

    return out;
}
