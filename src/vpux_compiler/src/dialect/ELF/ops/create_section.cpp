//
// Copyright 2022 Intel Corporation.
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

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELF/ops.hpp"

void vpux::ELF::CreateSectionOp::serialize(elf::Writer& writer, vpux::ELF::SectionMapType& sectionMap,
                                           vpux::ELF::SymbolMapType& symbolMap) {
    VPUX_UNUSED(symbolMap);
    auto section = writer.addBinaryDataSection<uint8_t>();

    const auto name = secName().str();
    section->setName(name);

    section->maskFlags(static_cast<elf::Elf_Xword>(secFlags()));
    section->setAddrAlign(secAddrAlign());

    auto block = getBody();
    for (auto& op : block->getOperations()) {
        if (op.hasTrait<vpux::ELF::BinaryOpInterface::Trait>()) {
            auto binaryOp = llvm::cast<vpux::ELF::BinaryOpInterface>(op);

            binaryOp.serialize(*section);
        }
    }

    sectionMap[getOperation()] = section;
}
