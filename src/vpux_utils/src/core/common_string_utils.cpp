//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/core/common_string_utils.hpp"

#include <algorithm>
#include <cstdarg>
#include <stdexcept>

namespace vpux {

//
// splitStringList
//

void splitStringList(const std::string& str, char delim, std::function<void(std::string_view)> callback) {
    const auto begin = str.begin();
    const auto end = str.end();

    auto curBegin = begin;
    auto curEnd = begin;
    while (curEnd != end) {
        while (curEnd != end && *curEnd != delim) {
            ++curEnd;
        }

        callback(std::string_view(&(*curBegin), static_cast<size_t>(curEnd - curBegin)));

        if (curEnd != end) {
            ++curEnd;
            curBegin = curEnd;
        }
    }
}

//
// removeCharFromString
//

std::string removeCharFromString(std::string&& str, char ch) {
    str.erase(std::remove(str.begin(), str.end(), ch), str.end());
    return std::move(str);
}

//
// eraseSubString
//

std::string eraseSubString(std::string&& str, const std::string& strToRemove, bool removeAllAfterSubstr) {
    const auto pos = str.find(strToRemove);

    if (pos != std::string::npos) {
        if (removeAllAfterSubstr) {
            str.erase(pos);
        } else {
            str.erase(pos, strToRemove.size());
        }
    }

    return std::move(str);
}

std::string printFormattedCStr(const char* fmt, ...) {
    std::va_list ap;
    va_start(ap, fmt);
    std::va_list apCopy;
    va_copy(apCopy, ap);
    const auto requiredBytes = vsnprintf(nullptr, 0, fmt, ap);
    va_end(ap);
    if (requiredBytes < 0) {
        va_end(apCopy);
        throw std::runtime_error(std::string("vsnprintf got error: ") + strerror(errno) + ", fmt: " + fmt);
    }
    std::string out(requiredBytes, 0);  // +1 implicitly
    vsnprintf(out.data(), requiredBytes + 1, fmt, apCopy);
    va_end(apCopy);
    return out;
}
}  // namespace vpux
