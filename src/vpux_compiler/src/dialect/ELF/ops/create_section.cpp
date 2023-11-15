//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELF/ops.hpp"

void vpux::ELF::CreateSectionOp::serialize(elf::Writer& writer, vpux::ELF::SectionMapType& sectionMap,
                                           vpux::ELF::SymbolMapType& symbolMap) {
    VPUX_UNUSED(symbolMap);
    const auto name = getSecName().str();
    auto section = writer.addBinaryDataSection<uint8_t>(name);
    section->maskFlags(static_cast<elf::Elf_Xword>(getSecFlags()));
    section->setAddrAlign(getSecAddrAlign());

    auto block = getBody();
    for (auto& op : block->getOperations()) {
        if (op.hasTrait<vpux::ELF::BinaryOpInterface::Trait>()) {
            auto binaryOp = llvm::cast<vpux::ELF::BinaryOpInterface>(op);

            binaryOp.serialize(*section);
        }
    }

    sectionMap[getOperation()] = section;
}
