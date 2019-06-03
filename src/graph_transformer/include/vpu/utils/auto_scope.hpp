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

#include <functional>

#include <details/ie_exception.hpp>

namespace vpu {

class AutoScope final {
public:
    explicit AutoScope(const std::function<void()>& func) : _func(func) {}

    ~AutoScope() {
        if (_func != nullptr) {
            _func();
        }
    }

    void callAndRelease() {
        if (_func != nullptr) {
            _func();
            _func = nullptr;
        }
    }

    void release() {
        _func = nullptr;
    }

    AutoScope(const AutoScope& other) = delete;
    AutoScope& operator=(const AutoScope&) = delete;

private:
    std::function<void()> _func;
};

}  // namespace vpu
