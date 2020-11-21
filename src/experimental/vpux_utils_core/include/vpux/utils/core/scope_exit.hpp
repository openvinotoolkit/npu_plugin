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
// RAII wrapper for callback to call it at the end of current scope.
//

#pragma once

#include <llvm/ADT/ScopeExit.h>

#include <utility>

namespace vpux {

using llvm::make_scope_exit;

namespace details {

struct ScopeExitTag final {};

template <class Func>
auto operator<<(ScopeExitTag, Func&& func) {
    return make_scope_exit(std::forward<Func>(func));
}

}  // namespace details

#define VPUX_SCOPE_EXIT const auto VPUX_UNIQUE_NAME(scopeExit) = vpux::details::ScopeExitTag{} << [&]()

}  // namespace vpux
