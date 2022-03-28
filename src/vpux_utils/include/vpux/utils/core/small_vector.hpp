//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

//
// `std::vector` like container with pre-allocated buffer for small sizes.
//

#pragma once

#include <llvm/ADT/SmallVector.h>

namespace vpux {

using llvm::SmallVector;
using llvm::SmallVectorImpl;

}  // namespace vpux
