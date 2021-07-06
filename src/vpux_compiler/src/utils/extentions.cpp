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

#include "vpux/compiler/utils/extentions.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>

//
// mlir::Value
//

mlir::Operation* vpux::getFirstUser(mlir::Value output) {
    VPUX_THROW_UNLESS(output != nullptr, "Got NULL pointer in getFirstUser");

    const auto& users = output.getUsers();
    const auto firstUser = std::min_element(users.begin(), users.end(), [](const auto& lhs, const auto& rhs) {
        return lhs->isBeforeInBlock(rhs);
    });

    return firstUser == users.end() ? nullptr : *firstUser;
}

//
// DataOrderInfo
//

void vpux::fillDataInfo(DataOrderInfo& info, size_t inNum, size_t outNum, const DimsOrder& mainOrder) {
    for (size_t i = 0; i < inNum; ++i) {
        info.setInput(i, mainOrder);
    }

    for (size_t i = 0; i < outNum; ++i) {
        info.setOutput(i, mainOrder);
    }
}
