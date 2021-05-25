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
