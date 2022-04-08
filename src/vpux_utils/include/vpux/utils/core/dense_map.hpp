//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/format.hpp"

#include <llvm/ADT/DenseMap.h>

namespace vpux {

using llvm::DenseMap;

}  // namespace vpux

//
// llvm::format_provider specialization
//

namespace llvm {

template <typename KeyT, typename ValueT, typename KeyInfoT, typename BucketT>
struct format_provider<DenseMap<KeyT, ValueT, KeyInfoT, BucketT>> final : vpux::MapFormatProvider {};

}  // namespace llvm
