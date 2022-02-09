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

#include "vpux/compiler/dialect/ELF/attributes.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/stl_extras.hpp"

#include <vpux_elf/writer.hpp>

namespace vpux {
namespace ELF {

typedef llvm::DenseMap<mlir::Operation*, elf::writer::Section*> SectionMapType;
// Note that this works since in our case the IR is immutable troughout the life-time of the map.
typedef llvm::DenseMap<mlir::Operation*, elf::writer::Symbol*> SymbolMapType;

}  // namespace ELF
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/ELF/generated/ops_interfaces.hpp.inc>
