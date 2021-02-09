//
// Copyright 2020 Intel Corporation.
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

//
// Light-weight non-owning wrapper for callback functions.
//
// Can be used to pass some callback to another function in case if the callback
// wo't be stored for future use.
//

#pragma once

#include <llvm/ADT/STLExtras.h>

namespace vpux {

template <typename Fn>
using FuncRef = llvm::function_ref<Fn>;

}  // namespace vpux
