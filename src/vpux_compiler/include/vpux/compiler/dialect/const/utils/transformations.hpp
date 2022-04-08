//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/dialect/const/attributes/content.hpp"

namespace vpux {
namespace Const {
namespace details {

//
// memPermuteTransformation
//

vpux::Const::Content memPermuteTransformation(vpux::Const::Content& input, vpux::NDTypeInterface outType,
                                              mlir::AffineMap memPerm);

}  // namespace details
}  // namespace Const
}  // namespace vpux
