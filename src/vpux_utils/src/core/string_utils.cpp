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
