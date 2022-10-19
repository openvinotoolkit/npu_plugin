//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/ELF/attributes.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/stl_extras.hpp"

#include "vpux/utils/core/dense_map.hpp"

#include <vpux_elf/writer.hpp>

namespace vpux {
namespace ELF {

typedef DenseMap<mlir::Operation*, elf::writer::Section*> SectionMapType;
// Note that this works since in our case the IR is immutable troughout the life-time of the map.
typedef DenseMap<mlir::Operation*, elf::writer::Symbol*> SymbolMapType;

//
// ElfSectionInterface
//

template <typename ConcreteOp>
mlir::Block* getSectionBlock(ConcreteOp op) {
    mlir::Operation* operation = op.getOperation();
    auto& region = operation->getRegion(0);

    if (region.empty()) {
        region.emplaceBlock();
    }

    return &region.front();
}

}  // namespace ELF
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/ELF/generated/ops_interfaces.hpp.inc>
