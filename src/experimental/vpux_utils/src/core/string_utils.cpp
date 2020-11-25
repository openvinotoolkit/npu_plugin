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
