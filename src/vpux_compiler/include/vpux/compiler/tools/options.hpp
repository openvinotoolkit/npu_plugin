//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/utils/core/string_ref.hpp"

namespace vpux {

vpux::VPU::ArchKind parseArchKind(int argc, char* argv[], StringRef helpHeader = "");

}  // namespace vpux
