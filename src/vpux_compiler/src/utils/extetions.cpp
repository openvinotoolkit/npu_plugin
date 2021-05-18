//
// Copyright Intel Corporation.
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
