//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUMI37XX/attributes.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// Generated
//

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/VPUMI37XX/generated/attributes.cpp.inc>
#undef GET_ATTRDEF_CLASSES

//
// Dialect hooks
//

void vpux::VPUMI37XX::VPUMI37XXDialect::registerAttributes() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include <vpux/compiler/dialect/VPUMI37XX/generated/attributes.cpp.inc>
            >();
}
