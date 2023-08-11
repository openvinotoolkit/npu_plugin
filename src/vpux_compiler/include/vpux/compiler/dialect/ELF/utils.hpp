//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>
#include "vpux/compiler/act_kernels/nce2p7.h"
#include "vpux/compiler/dialect/ELF/ops.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux;

namespace llvm {
using RelocKey = std::pair<mlir::Value, ELF::CreateSymbolTableSectionOp>;
template <>
struct DenseMapInfo<RelocKey> {
    static RelocKey getEmptyKey() {
        void* pointer = llvm::DenseMapInfo<void*>::getEmptyKey();
        return RelocKey(RelocKey::first_type::getFromOpaquePointer(pointer),
                        RelocKey::second_type::getFromOpaquePointer(pointer));
    }

    static RelocKey getTombstoneKey() {
        void* pointer = llvm::DenseMapInfo<void*>::getTombstoneKey();
        return RelocKey(RelocKey::first_type::getFromOpaquePointer(pointer),
                        RelocKey::second_type::getFromOpaquePointer(pointer));
    }

    static unsigned getHashValue(RelocKey val) {
        auto h1 = hash_value(val.first.getAsOpaquePointer());
        auto h2 = hash_value(val.second.getAsOpaquePointer());

        return h1 * h2;
    }

    static bool isEqual(RelocKey lhs, RelocKey rhs) {
        auto l1 = DenseMapInfo<mlir::Value>::isEqual(lhs.first, rhs.first);
        auto l2 = DenseMapInfo<mlir::Operation*>::isEqual(lhs.second.getOperation(), rhs.second.getOperation());

        return l1 && l2;
    }
};
}  // namespace llvm

namespace vpux {

// creates a linear (1D) MemrefType of dimension (memrefSize x dataType)
mlir::MemRefType getLinearMemrefType(mlir::MLIRContext* ctx, int64_t memrefSize, mlir::Type dataType,
                                     VPU::MemoryKind memKind);

namespace ELF {

std::pair<const uint8_t*, size_t> getDataAndSizeOfElfSection(const std::vector<uint8_t>& elfBlob,
                                                             const std::vector<std::string> possibleSecNames);

size_t getOffsetOfOpInSection(mlir::Value& op, mlir::Value& section);
size_t getOffsetOfOpInSection(mlir::Value& op);

SmallString getSwKernelArchString(VPU::ArchKind archKind);

class RelocationManager {
public:
    RelocationManager() = default;

    RelocationManager(mlir::func::FuncOp funcOp): funcOp_(funcOp) {
    }

    void init(mlir::func::FuncOp funcOp);
    void initCMXSymTab(ELF::CreateSymbolTableSectionOp cmxMappingSymTab);
    ELF::ElfSectionInterface getSection(mlir::Value value);
    ELF::CreateSymbolTableSectionOp getCMXSymTab();
    ELF::CreateSymbolTableSectionOp getSymTab(mlir::Value value);
    ELF::CreateRelocationSectionOp getRelocSection(ELF::ElfSectionInterface section,
                                                   ELF::CreateSymbolTableSectionOp symTab);
    static ELF::SymbolOp getSymbol(ELF::ElfSectionInterface section);

private:
    mlir::func::FuncOp funcOp_ = nullptr;
    ELF::CreateSymbolTableSectionOp cmxMappingSymTab_ = nullptr;
    llvm::DenseMap<mlir::Value, ELF::ElfSectionInterface> sectionMap_;
    llvm::DenseMap<mlir::Value, ELF::CreateSymbolTableSectionOp> symTabMap_;
    llvm::DenseMap<std::pair<mlir::Value, ELF::CreateSymbolTableSectionOp>, ELF::CreateRelocationSectionOp> relocMap_;
};

namespace math {

size_t gcd(size_t a, size_t b);
size_t lcm(size_t a, size_t b);

}  // namespace math

constexpr size_t VPUX_SHAVE_ALIGNMENT = Byte(1_KB).count();
constexpr size_t VPUX_DEFAULT_ALIGNMENT = (64_Byte).count();
constexpr size_t VPUX_NO_ALIGNMENT = (1_Byte).count();

}  // namespace ELF
}  // namespace vpux
