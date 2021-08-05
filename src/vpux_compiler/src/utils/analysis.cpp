//
// Copyright Intel Corporation.
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

#include "vpux/compiler/utils/analysis.hpp"

#include "vpux/utils/core/error.hpp"

#include <mlir/IR/Operation.h>

#include <algorithm>

using namespace vpux;

//
// getFirstUser
//

mlir::Operation* vpux::getFirstUser(mlir::Value output) {
    VPUX_THROW_UNLESS(output != nullptr, "Got NULL pointer in getFirstUser");

    const auto users = output.getUsers();
    const auto firstUser = std::min_element(users.begin(), users.end(), [](mlir::Operation* lhs, mlir::Operation* rhs) {
        return lhs->getBlock() == rhs->getBlock() && lhs->isBeforeInBlock(rhs);
    });

    return firstUser == users.end() ? nullptr : *firstUser;
}
