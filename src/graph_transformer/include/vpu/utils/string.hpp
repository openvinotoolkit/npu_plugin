//
// Copyright (C) 2018-2019 Intel Corporation.
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

#pragma once

#include <string>
#include <vector>
#include <set>
#include <unordered_set>
#include <sstream>
#include <utility>

#include <details/caseless.hpp>

#include <vpu/utils/containers.hpp>

namespace vpu {

namespace ie = InferenceEngine;

namespace impl {

inline void insertToContainer(std::vector<std::string>& cont, std::string&& val) {
    cont.emplace_back(val);
}

template <int Capacity>
void insertToContainer(SmallVector<std::string, Capacity>& cont, std::string&& val) {
    cont.emplace_back(val);
}

inline void insertToContainer(std::set<std::string>& cont, std::string&& val) {
    cont.emplace(val);
}

inline void insertToContainer(std::unordered_set<std::string>& cont, std::string&& val) {
    cont.emplace(val);
}

inline void insertToContainer(ie::details::caseless_set<std::string>& cont, std::string&& val) {
    cont.emplace(val);
}

}  // namespace impl

template <class Cont>
void splitStringList(const std::string& str, Cont& out, char delim) {
    out.clear();

    if (str.empty())
        return;

    std::istringstream istr(str);

    std::string elem;
    while (std::getline(istr, elem, delim)) {
        if (elem.empty()) {
            continue;
        }

        impl::insertToContainer(out, std::move(elem));
    }
}

}  // namespace vpu
