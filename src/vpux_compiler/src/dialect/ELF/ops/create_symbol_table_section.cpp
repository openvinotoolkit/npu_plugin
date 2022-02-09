#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/ELF/ops.hpp"

void vpux::ELF::CreateSymbolTableSectionOp::serialize(elf::Writer& writer, vpux::ELF::SectionMapType& sectionMap,
                                                      vpux::ELF::SymbolMapType& symbolMap) {
    const auto name = secName().str();
    auto section = writer.addSymbolSection(name);
    section->maskFlags(static_cast<elf::Elf_Xword>(secFlags()));

    auto block = getBody();
    for (auto& op : block->getOperations()) {
        auto symbol = section->addSymbolEntry();

        if (auto symOp = llvm::dyn_cast<vpux::ELF::SymbolOp>(op)) {
            symOp.serialize(symbol, sectionMap);
            symbolMap[symOp.getOperation()] = symbol;
        } else if (auto placeholder = llvm::dyn_cast<vpux::ELF::PutOpInSectionOp>(op)) {
            auto actualOp = placeholder.inputArg().getDefiningOp();
            auto symOp = llvm::dyn_cast<vpux::ELF::SymbolOp>(actualOp);

            VPUX_THROW_UNLESS(symOp != nullptr,
                              "Symbol table section op is expected to have placeholders that point to SymbolOps only."
                              " Got *actualOp {0}.",
                              *actualOp);

            symOp.serialize(symbol, sectionMap);
            symbolMap[symOp.getOperation()] = symbol;
        } else {
            VPUX_THROW("Symbol table section op is expected to have either SymbolOps or placeholders that point to "
                       "SymbolOps. Got {0}.",
                       op);
        }
    }

    sectionMap[getOperation()] = section;
}
