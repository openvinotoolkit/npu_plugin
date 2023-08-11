//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/utils/custom_pwl_table.hpp"

namespace vpux {
Optional<vpux::PWLTableEntry> getPWLEntryForAlpha0(const int64_t zeroPoint);
Optional<vpux::PWLTableEntry> getPWLEntryForAlpha1(const int64_t zeroPoint);
Optional<vpux::PWLTableEntry> getPWLEntryForAlpha2(const int64_t zeroPoint);
Optional<vpux::PWLTableEntry> getPWLEntryForAlpha25(const int64_t zeroPoint);
}  // namespace vpux
