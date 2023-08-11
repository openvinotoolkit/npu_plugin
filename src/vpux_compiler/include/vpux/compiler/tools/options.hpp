//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/attributes.hpp"

namespace vpux {

vpux::VPU::ArchKind parseArchKind(int argc, char* argv[]);

}  // namespace vpux
