//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/ELF/utils.hpp"
#include <vpux_elf/accessor.hpp>
#include <vpux_elf/reader.hpp>
#include "vpux/compiler/dialect/VPURT/ops.hpp"

std::pair<const uint8_t*, size_t> vpux::ELF::getDataAndSizeOfElfSection(
        const std::vector<uint8_t>& elfBlob, const std::vector<std::string> possibleSecNames) {
    auto accessor = elf::ElfDDRAccessManager(elfBlob.data(), elfBlob.size());
    auto elf_reader = elf::Reader<elf::ELF_Bitness::Elf32>(&accessor);

    const uint8_t* secData = nullptr;
    uint32_t secSize = 0;

    bool secFound = false;

    for (size_t i = 0; i < elf_reader.getSectionsNum(); ++i) {
        auto section = elf_reader.getSection(i);
        const auto secName = section.getName();
        const auto sectionHeader = section.getHeader();

        for (auto possibleSecName : possibleSecNames) {
            if (strcmp(secName, possibleSecName.c_str()) == 0) {
                secSize = sectionHeader->sh_size;
                secData = section.getData<uint8_t>();
                secFound = true;
                break;
            }
        }
    }
    VPUX_THROW_UNLESS(secFound, "Section {0} not found in ELF", possibleSecNames);

    return {secData, secSize};
}

size_t vpux::ELF::getOffsetOfOpInSection(mlir::Value& op, mlir::Value& section) {
    size_t totalOffset = 0;

    // specific case of uninitialized buffer where the offset to it is actually specified as an op attribute and there
    // is no need for any computation
    auto actualOp = op.getDefiningOp();
    if (auto declareBufferOp = mlir::dyn_cast<vpux::VPURT::DeclareBufferOp>(actualOp)) {
        return declareBufferOp.getByteOffset();
    }

    auto elfSection = mlir::cast<vpux::ELF::ElfSectionInterface>(section.getDefiningOp());

    bool opFound = false;

    for (auto& inner_op : elfSection.getBlock()->getOperations()) {
        if (mlir::isa<vpux::ELF::BinaryOpInterface>(inner_op)) {
            if (auto putOpInSectionOp = mlir::dyn_cast<vpux::ELF::PutOpInSectionOp>(inner_op)) {
                if (putOpInSectionOp.getInputArg() == op.getDefiningOp()->getResult(0)) {
                    opFound = true;
                    break;
                }
                auto binaryOp =
                        mlir::cast<vpux::ELF::BinaryOpInterface>(putOpInSectionOp.getInputArg().getDefiningOp());
                totalOffset += binaryOp.getBinarySize();

            } else {
                if (inner_op.getResult(0) == op.getDefiningOp()->getResult(0)) {
                    opFound = true;
                    break;
                }
                auto binaryOp = mlir::cast<vpux::ELF::BinaryOpInterface>(inner_op);
                totalOffset += binaryOp.getBinarySize();
            }
        }
    }

    VPUX_THROW_UNLESS(opFound, "Offset can't be computed because op was not found in section");

    return totalOffset;
}

size_t vpux::ELF::getOffsetOfOpInSection(mlir::Value& op) {
    auto actualOp = op.getDefiningOp();
    auto declareBufferOp = mlir::dyn_cast<vpux::VPURT::DeclareBufferOp>(actualOp);
    VPUX_THROW_UNLESS(
            declareBufferOp != nullptr && declareBufferOp.getMemorySpace() == vpux::VPURT::BufferSection::CMX_NN,
            "This version of getOffsetOfOpInSection() only works with CMX Buffers");

    auto type = op.getType().dyn_cast<vpux::NDTypeInterface>();

    auto tile = type.getMemSpace().getIndex().value_or(0);

    return tile * mvds::nce2p7::CMX_SLICE_SIZE + declareBufferOp.getByteOffset();
}

SmallString vpux::ELF::getSwKernelArchString(VPU::ArchKind archKind) {
    VPUX_THROW_UNLESS(archKind == VPU::ArchKind::VPUX37XX, "The only supported architecture for sw kernels is 3720xx");
    return SmallString("3720xx");
}

void vpux::ELF::RelocationManager::init(mlir::func::FuncOp funcOp) {
    funcOp_ = funcOp;
}

void vpux::ELF::RelocationManager::initCMXSymTab(ELF::CreateSymbolTableSectionOp cmxMappingSymTab) {
    VPUX_THROW_WHEN(cmxMappingSymTab_ != nullptr, "CMX Mapping SymTab is already init!");
    cmxMappingSymTab_ = cmxMappingSymTab;
}

ELF::ElfSectionInterface vpux::ELF::RelocationManager::getSection(mlir::Value value) {
    VPUX_THROW_UNLESS(funcOp_ != nullptr, "Relocation Manager is not initialized with a mlir::func::FuncOp");

    auto section = sectionMap_.find(value);
    if (section != sectionMap_.end()) {
        return section->second;
    }

    auto parent = value.getDefiningOp();
    if (parent->hasTrait<ELF::ElfSectionInterface::Trait>()) {
        sectionMap_[value] = mlir::cast<ELF::ElfSectionInterface>(parent);
        return parent;
    }

    auto users = value.getUsers();
    ELF::PutOpInSectionOp placer;
    for (auto user : users) {
        if (mlir::isa<ELF::PutOpInSectionOp>(user)) {
            parent = user->getParentOp();
            sectionMap_[value] = mlir::cast<ELF::ElfSectionInterface>(parent);
            return parent;
        }
    }

    VPUX_THROW("Cannot get the section of value {0}", value.getDefiningOp()->getName());
    return nullptr;
}

ELF::CreateSymbolTableSectionOp vpux::ELF::RelocationManager::getCMXSymTab() {
    VPUX_THROW_UNLESS(cmxMappingSymTab_ != nullptr, "Relocation Manager: CMX Mapping SymTab not init!");
    return cmxMappingSymTab_;
}

ELF::CreateSymbolTableSectionOp vpux::ELF::RelocationManager::getSymTab(mlir::Value value) {
    VPUX_THROW_UNLESS(funcOp_ != nullptr, "Relocation Manager is not initialized with a mlir::func::FuncOp");

    auto binaryOp = mlir::dyn_cast<ELF::BinaryOpInterface>(value.getDefiningOp());

    if (binaryOp != nullptr) {
        auto memSpace = binaryOp.getMemorySpace();
        if (memSpace == VPURT::BufferSection::CMX_NN || memSpace == VPURT::BufferSection::Register) {
            VPUX_THROW_UNLESS(cmxMappingSymTab_ != nullptr, "RelocManager: cmxMappingSymTab is not init!");
            symTabMap_[value] = cmxMappingSymTab_;
            return cmxMappingSymTab_;
        }
    }

    auto section = symTabMap_.find(value);
    if (section != symTabMap_.end()) {
        return section->second;
    }

    auto lookupSymTab = [&](mlir::Value::user_range children) -> ELF::CreateSymbolTableSectionOp {
        for (auto child : children) {
            if (mlir::isa<ELF::SymbolOp>(child)) {
                if (mlir::isa<ELF::CreateSymbolTableSectionOp>(child->getParentOp())) {
                    auto symTab = mlir::cast<ELF::CreateSymbolTableSectionOp>(child->getParentOp());
                    symTabMap_[value] = symTab;
                    return symTab;
                }

                auto grandChildren = child->getUsers();
                for (auto grandChild : grandChildren) {
                    if (mlir::isa<ELF::PutOpInSectionOp>(grandChild)) {
                        auto symTab = mlir::cast<ELF::CreateSymbolTableSectionOp>(grandChild->getParentOp());
                        symTabMap_[value] = symTab;
                        return symTab;
                    }
                }
                VPUX_THROW("Symbol at {0} not in a section", child->getLoc());
            }
        }
        return nullptr;
    };

    auto children = value.getUsers();
    auto symTab = lookupSymTab(children);

    if (symTab) {
        return symTab;
    }

    // if OP has no symbol associated, then we will look at the Symtab that defines it's section symbol
    auto containerSection = getSection(value);
    children = containerSection.getOperation()->getResult(0).getUsers();
    symTab = lookupSymTab(children);

    if (symTab) {
        return symTab;
    }

    VPUX_THROW("Could not find symtab for op {0} at {1}", value.getDefiningOp()->getName(),
               value.getDefiningOp()->getLoc());

    return nullptr;
}

ELF::CreateRelocationSectionOp vpux::ELF::RelocationManager::getRelocSection(ELF::ElfSectionInterface section,
                                                                             ELF::CreateSymbolTableSectionOp symTab) {
    VPUX_THROW_UNLESS(funcOp_ != nullptr, "Relocation Manager is not initialized with a mlir::func::FuncOp");

    // for some reason, can't construct an interface object from opaque pointer. So will use the result as the key
    // for the map
    auto result = section.getOperation()->getResult(0);

    auto relocSection = relocMap_.find(std::make_pair(result, symTab));
    if (relocSection != relocMap_.end()) {
        return relocSection->second;
    }

    auto builder = mlir::OpBuilder::atBlockTerminator(&funcOp_.getBody().front());

    std::string secName = std::string(".rlt") + section.getName().str();
    auto newRelocation = builder.create<ELF::CreateRelocationSectionOp>(
            section.getLoc(),
            vpux::ELF::SectionType::get(builder.getContext()),  // mlir::Type
            llvm::StringRef(secName),                           // llvm::StringRef secName
            symTab.getResult(),                                 // sourceSymbolTalbeSection
            section.getOperation()->getResult(0),               // targetSection
            vpux::ELF::SectionFlagsAttr::SHF_INFO_LINK);

    relocMap_[std::make_pair(result, symTab)] = newRelocation;
    return newRelocation;
}

// TODO:
// create non-static getSymbol() method that returns a (generic or section) symbol for any mlir:Value (if it exists)
ELF::SymbolOp vpux::ELF::RelocationManager::getSymbol(ELF::ElfSectionInterface section) {
    mlir::Operation* op = section.getOperation();
    for (auto user : op->getUsers()) {
        if (mlir::isa<ELF::SymbolOp>(user)) {
            return mlir::cast<ELF::SymbolOp>(user);
        }
    }

    VPUX_THROW("No symbol op defined for section {0} at {1}", op->getName(), op->getLoc());
    return nullptr;
}

mlir::MemRefType vpux::getLinearMemrefType(mlir::MLIRContext* ctx, int64_t memrefSize, mlir::Type dataType,
                                           VPU::MemoryKind memKind) {
    VPUX_THROW_UNLESS(dataType.isIntOrFloat(), "Data Type of the MemRef must be an Integer or Float Type");

    const auto memrefShape = SmallVector<int64_t>{memrefSize};
    auto memKindAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(memKind));
    const auto memKindSymbolAttr = vpux::IndexedSymbolAttr::get(ctx, memKindAttr);
    unsigned int perm[1] = {0};
    auto map = mlir::AffineMap::getPermutationMap(to_small_vector(perm), ctx);

    auto memrefType = mlir::MemRefType::get(memrefShape, dataType, map, memKindSymbolAttr);
    return memrefType;
}

size_t vpux::ELF::math::gcd(size_t a, size_t b) {
    if (b == 0) {
        return a;
    }
    return gcd(b, a % b);
}

size_t vpux::ELF::math::lcm(size_t a, size_t b) {
    return (a / vpux::ELF::math::gcd(a, b)) * b;
}
