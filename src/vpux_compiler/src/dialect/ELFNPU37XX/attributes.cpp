//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/ELFNPU37XX/attributes.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/dialect.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// Generated
//

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/ELFNPU37XX/attributes.cpp.inc>

#include <vpux/compiler/dialect/ELFNPU37XX/enums.cpp.inc>

//
// Dialect hooks
//

void ELFNPU37XX::ELFNPU37XXDialect::registerAttributes() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include <vpux/compiler/dialect/ELFNPU37XX/attributes.cpp.inc>
            >();
}
