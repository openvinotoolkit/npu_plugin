#include <vpux_elf/writer.hpp>
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
