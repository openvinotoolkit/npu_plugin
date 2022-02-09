#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELF/ops.hpp"

void vpux::ELF::CreateSectionOp::serialize(elf::Writer& writer, vpux::ELF::SectionMapType& sectionMap,
                                           vpux::ELF::SymbolMapType& symbolMap) {
    VPUX_UNUSED(symbolMap);
    const auto name = secName().str();
    auto section = writer.addBinaryDataSection<uint8_t>(name);
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
