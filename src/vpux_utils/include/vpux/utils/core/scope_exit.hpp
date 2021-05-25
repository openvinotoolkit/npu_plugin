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
