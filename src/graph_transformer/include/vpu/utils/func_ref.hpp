//
// Copyright 2019 Intel Corporation.
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

#include <utility>
#include <type_traits>

namespace vpu {

//
// Non-owning alternative for std::function
//

template <typename> class FuncRef;

template <typename R, typename... Args>
class FuncRef<R(Args...)> {
public:
    template <class Func>
    FuncRef(const Func& func) :
            _realFuncPtr(&func),
            _impl(&caller<typename std::remove_reference<Func>::type>) {
        using actual_result_type = typename std::result_of<Func(Args...)>::type;
        static_assert(
            !std::is_reference<R>::value || std::is_reference<actual_result_type>::value,
            "Mismatch between Func and FuncRef prototype");
    }

    R operator()(Args... args) const {
        return _impl(_realFuncPtr, std::forward<Args>(args)...);
    }

private:
    template <class Func>
    static R caller(const void* realFuncPtr, Args... args) {
        const auto& realFunc = *static_cast<const Func*>(realFuncPtr);
        return realFunc(std::forward<Args>(args)...);
    }

private:
    using ImplFunc = R(*)(const void*, Args...);

    const void* _realFuncPtr = nullptr;
    ImplFunc _impl = nullptr;
};

}  // namespace vpu
