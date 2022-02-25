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

namespace {
mlir::Operation* getParentSectionOp(mlir::Value val) {
    // If one of the users of the value is a PutOpInSection op, then we are interested in its encapsulating section.
    // Based on IR validity, any PutOpInSection op is always in a CreateSymbolTableSection (or CreateSection).
    mlir::Operation* op = nullptr;
    for (auto user : val.getUsers()) {
        if (mlir::dyn_cast_or_null<vpux::ELF::PutOpInSectionOp>(user)) {
            op = user;
            // In a section we should have only 1 PutOpInSectionOp user for each object put in it.
            break;
        }
    }

    // This happens normally in case val is a BlockArgument (e.g., "arg0" in the .mlir file).
    //  OR in the case of a value assigned in the FuncOp scope.
    if (op == nullptr) {
        op = val.getDefiningOp();

        return val.getParentRegion()->getParentOp();
    }
    VPUX_THROW_UNLESS(op != nullptr, "Both user and producer operation can't be found.");

    auto region = op->getParentRegion();
    VPUX_THROW_UNLESS(region != nullptr, "Unlinked ops are unsupported");

    return region->getParentOp();
}

}  // namespace

void vpux::ELF::SymbolOp::serialize(elf::writer::Symbol* symbol, vpux::ELF::SectionMapType& sectionMap) {
    
    if (isBuiltin())
        return;
    
    auto symName = name().getValueOr("");
    auto symType = type().getValueOr(vpux::ELF::SymbolTypeAttr::STT_NOTYPE);
    auto symSize = size().getValueOr(0);
    auto symVal = value().getValueOr(0);

    /* From serialization perspective symbols can be of 5 types:
        - SECTION symbols : in this case the parentSection is the defining op itself;
        - Generic symbols : Symbols representing an OP inside the IR. In this case we need the parent section of either
       the OP or it's placeholder;
        - I/O symbols : symbols that represent function arguments. In this case we will not have a parentSection, and no
       relatedSection;
        - Symbols refering to the "Special Symbol Table";
        - Standalone symbols : symbols that do not relate to any entity inside the IR (nor the ELF itself).
      The ticket #29144 plans to handle these last 2 types of sections.
    */

    // We initialize parentSection to nullptr, since inputArg() can be a BlockArgument,
    //   which has getDefiningOp() equal to nullptr.
    mlir::Operation* parentSection = nullptr;

    if (auto inputArgDefOp = inputArg().getDefiningOp()) {
        if (mlir::isa<ELF::ElfSectionInterface>(inputArgDefOp)) {
            parentSection = inputArgDefOp;
        } else {
            parentSection = getParentSectionOp(inputArg());

            if (mlir::isa<mlir::FuncOp>(parentSection)) {
                parentSection = nullptr;
            } else {
                VPUX_THROW_UNLESS(parentSection != nullptr, "Could not find valid parent section for op");
            }
        }
    }

    symbol->setName(symName.str());
    symbol->setType(static_cast<elf::Elf_Word>(symType));
    symbol->setSize(symSize);
    symbol->setValue(symVal);

    if (parentSection != nullptr) {
        auto sectionMapEntry = sectionMap.find(parentSection);
        VPUX_THROW_UNLESS(sectionMapEntry != sectionMap.end(), "Unable to find section entry for SymbolOp");
        auto sectionEntry = sectionMapEntry->second;

        symbol->setRelatedSection(sectionEntry);
    }
}
