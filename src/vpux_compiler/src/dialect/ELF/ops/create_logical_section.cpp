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

#include <elf/writer.hpp>
#include "vpux/compiler/dialect/ELF/ops.hpp"

void vpux::ELF::CreateLogicalSectionOp::serialize(elf::Writer& writer, vpux::ELF::SectionMapType& sectionMap,
                                                  vpux::ELF::SymbolMapType& symbolMap) {
    VPUX_UNUSED(symbolMap);
    const auto name = secName().str();
    auto section = writer.addEmptySection(name);
    section->maskFlags(static_cast<elf::Elf_Xword>(secFlags()));
    section->setAddrAlign(secAddrAlign());

    size_t totalSize = 0;
    auto block = getBody();
    for (auto& op : block->getOperations()) {
        if (op.hasTrait<vpux::ELF::BinaryOpInterface::Trait>()) {
            auto binaryOp = llvm::cast<vpux::ELF::BinaryOpInterface>(op);

            totalSize += binaryOp.getBinarySize();
        }
    }
    section->setSize(totalSize);

    sectionMap[getOperation()] = section;
}
