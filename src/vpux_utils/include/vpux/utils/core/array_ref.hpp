//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

//
// Non-owning wrapper for array range.
//

#pragma once

#include <llvm/ADT/ArrayRef.h>

namespace vpux {

using llvm::ArrayRef;
using llvm::makeArrayRef;

using llvm::makeMutableArrayRef;
using llvm::MutableArrayRef;

}  // namespace vpux
