//
// Copyright Intel Corporation.
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

#pragma once

#include "vpux/compiler/dialect/ELF/attributes/enums.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"

//#include <mlir/IR/BuiltinTypes.h>
//#include <mlir/IR/Operation.h>
//#include <mlir/IR/Value.h>

#include "vpux/compiler/utils/attributes.hpp"

//#include "vpux/utils/core/small_vector.hpp"

//#include <mlir/IR/OpDefinition.h>
//#include <mlir/IR/Operation.h>
//#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <elf/writer.hpp>

namespace vpux {
namespace VPUIPRegMapped {}  // namespace VPUIPRegMapped
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/ELF/generated/ops_interfaces.hpp.inc>
