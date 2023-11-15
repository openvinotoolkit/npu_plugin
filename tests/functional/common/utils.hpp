//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <openvino/runtime/core.hpp>

std::string getBackendName(const ov::Core& core);
std::vector<std::string> getAvailableDevices(const ov::Core& core);

template <typename T>
std::string vectorToString(std::vector<T> v) {
    std::ostringstream res;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i != 0) {
            res << ",";
        } else {
            res << "{";
        }
        res << v[i];
    }
    res << "}";
    return res.str();
}
