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

#pragma once

#include <mlir/IR/Value.h>
#include <vpux/compiler/conversion.hpp>

namespace vpux {

//
// mlir::Value
//

mlir::Operation* getFirstUser(mlir::Value output);

//
// DataOrderInfo
//

void fillDataInfo(DataOrderInfo& info, size_t inNum, size_t outNum, const DimsOrder& mainOrder);

}  // namespace vpux
