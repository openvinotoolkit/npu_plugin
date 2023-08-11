//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/ELF/utils.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

void vpux::VPURT::DeclareBufferOp::build(mlir::OpBuilder& builder, ::mlir::OperationState& state, mlir::Type type,
                                         VPURT::BufferSection section, int64_t byteOffset) {
    build(builder, state, type, VPURT::BufferSectionAttr::get(builder.getContext(), section), /*sectionIndex=*/nullptr,
          getIntAttr(builder, byteOffset), /*swizzlingKey=*/nullptr);
}

void vpux::VPURT::DeclareBufferOp::build(mlir::OpBuilder& builder, ::mlir::OperationState& state, mlir::Type type,
                                         VPURT::BufferSection section, ArrayRef<int64_t> sectionIndex,
                                         int64_t byteOffset) {
    build(builder, state, type, VPURT::BufferSectionAttr::get(builder.getContext(), section),
          getIntArrayAttr(builder, sectionIndex), getIntAttr(builder, byteOffset), /*swizzlingKey=*/nullptr);
}

void vpux::VPURT::DeclareBufferOp::build(mlir::OpBuilder& builder, ::mlir::OperationState& state, mlir::Type type,
                                         VPURT::BufferSection section, int64_t sectionIndex, int64_t byteOffset) {
    build(builder, state, type, VPURT::BufferSectionAttr::get(builder.getContext(), section),
          getIntArrayAttr(builder, makeArrayRef({sectionIndex})), getIntAttr(builder, byteOffset),
          /*swizzlingKey=*/nullptr);
}

void vpux::VPURT::DeclareBufferOp::build(mlir::OpBuilder& builder, ::mlir::OperationState& state, mlir::Type type,
                                         VPURT::BufferSection section, ArrayRef<int64_t> sectionIndex,
                                         int64_t byteOffset, int64_t swizzlingKey) {
    build(builder, state, type, VPURT::BufferSectionAttr::get(builder.getContext(), section),
          getIntArrayAttr(builder, sectionIndex), getIntAttr(builder, byteOffset), getIntAttr(builder, swizzlingKey));
}

void vpux::VPURT::DeclareBufferOp::build(mlir::OpBuilder& builder, ::mlir::OperationState& state, mlir::Type type,
                                         VPURT::BufferSection section, int64_t sectionIndex, int64_t byteOffset,
                                         int64_t swizzlingKey) {
    build(builder, state, type, VPURT::BufferSectionAttr::get(builder.getContext(), section),
          getIntArrayAttr(builder, makeArrayRef({sectionIndex})), getIntAttr(builder, byteOffset),
          getIntAttr(builder, swizzlingKey));
}

mlir::LogicalResult vpux::VPURT::DeclareBufferOp::verify() {
    const auto op = getOperation();
    const auto type = getType().cast<vpux::NDTypeInterface>();
    const auto opSection = section();

    if (!VPURT::isMemoryCompatible(opSection, type)) {
        return errorAt(op, "BufferSection '{0}' is not compatible with memory space '{1}'", opSection,
                       type.getMemSpace());
    }

    const auto maybeSectionIndex = sectionIndex();
    if (maybeSectionIndex.hasValue()) {
        if (maybeSectionIndex.getValue().empty()) {
            return errorAt(op, "Empty section index is not supported");
        }
    }

    if (section() == VPURT::BufferSection::CMX_NN) {
        const auto checkSectionIndex = [&op, &type](ArrayRef<int64_t> sectionIdx) {
            if (auto distributedType = type.dyn_cast<VPUIP::DistributedBufferType>()) {
                const auto distribution = distributedType.getDistribution();
                const auto numClusters = checked_cast<size_t>(distribution.num_clusters().getInt());
                if (numClusters != sectionIdx.size()) {
                    return errorAt(op, "Number of clusters '{0}' and section indexes '{1}' mismatch for op = {2}",
                                   numClusters, sectionIdx.size(), op);
                }
            }

            if (sectionIdx.size() == 1) {
                const auto memSpace = type.getMemSpace();
                if (memSpace == nullptr) {
                    return errorAt(op, "Output type must have CMX_NN memory space, op = {0}", op);
                }

                const auto maybeIdx = memSpace.getIndex();
                if (!maybeIdx.hasValue()) {
                    return errorAt(op,
                                   "Output type must have memory space index equal to '{0}', but it doesn't. declare "
                                   "buffer op = {1}",
                                   sectionIdx[0], op);
                }

                const auto memSpaceIdx = maybeIdx.getValue();
                if (memSpaceIdx != sectionIdx[0]) {
                    return errorAt(op, "Section index '{0}' and memory space index '{1}' mismatch for op = {2}",
                                   sectionIdx[0], memSpaceIdx, op);
                }
            }

            if (sectionIdx.size() > 1) {
                const auto distributedType = type.dyn_cast<VPUIP::DistributedBufferType>();
                const auto vpuipBufferType = type.dyn_cast<VPUIP::BufferType>();
                if (distributedType == nullptr && vpuipBufferType == nullptr) {
                    return errorAt(
                            op,
                            "Array of section indexes is supported only for vpuip/distributed buffer type, op = {0}",
                            op);
                }
            }

            return mlir::success();
        };

        if (!maybeSectionIndex.hasValue()) {
            if (!type.isa<VPUIP::DistributedBufferType>()) {
                return errorAt(op, "Section index is missing");
            }
        } else {
            const auto sectionIdx = parseIntArrayAttr<int64_t>(maybeSectionIndex.getValue());
            if (checkSectionIndex(sectionIdx).failed()) {
                return mlir::failure();
            }
        }
    } else if (section() == VPURT::BufferSection::DDR) {
        if (type.getMemSpace() == nullptr) {
            return errorAt(op, "Output type must have DDR memory space");
        }

        if (maybeSectionIndex.hasValue()) {
            const auto sectionIndex = parseIntArrayAttr<int64_t>(maybeSectionIndex.getValue());
            if (sectionIndex.size() == 1 && sectionIndex[0] != 0) {
                return errorAt(op, "Wrong section index value for DDR memory space: '{0}'", sectionIndex[0]);
            }

            if (sectionIndex.size() > 1) {
                return errorAt(op, "Array of section indexes is supported for DDR memory space");
            }
        }
    }

    return mlir::success();
}

SmallVector<int64_t> vpux::VPURT::DeclareBufferOp::getNonEmptySectionIndex() {
    if (sectionIndex().hasValue()) {
        return parseIntArrayAttr<int64_t>(sectionIndex().getValue());
    }
    return SmallVector<int64_t>({0});
}

void vpux::VPURT::DeclareBufferOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    VPUX_UNUSED(binDataSection);
}

size_t vpux::VPURT::DeclareBufferOp::getBinarySize() {
    const auto type = buffer().getType().cast<vpux::NDTypeInterface>();
    return type.getTotalAllocSize().count();
}

size_t vpux::VPURT::DeclareBufferOp::getAlignmentRequirements() {
    return ELF::VPUX_NO_ALIGNMENT;
}

vpux::VPURT::BufferSection vpux::VPURT::DeclareBufferOp::getMemorySpace() {
    return section();
}

vpux::ELF::SectionFlagsAttr vpux::VPURT::DeclareBufferOp::getAccessingProcs() {
    auto tempFlagsVal = vpux::ELF::SectionFlagsAttr::SHF_NONE;

    for (auto user : getResult().getUsers()) {
        if (auto binaryIface = mlir::dyn_cast<vpux::ELF::BinaryOpInterface>(user)) {
            tempFlagsVal = tempFlagsVal | binaryIface.getUserProcs();
        }
    }

    return tempFlagsVal;
}

vpux::ELF::SectionFlagsAttr vpux::VPURT::DeclareBufferOp::getUserProcs() {
    return (ELF::SectionFlagsAttr::SHF_NONE);
}
