//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPURT/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

void vpux::VPURT::DeclareBufferOp::build(mlir::OpBuilder& builder, ::mlir::OperationState& state, mlir::Type type,
                                         VPURT::BufferSection section, int64_t byteOffset) {
    build(builder, state, type, VPURT::BufferSectionAttr::get(builder.getContext(), section), /*sectionIndex=*/nullptr,
          getIntAttr(builder, byteOffset));
}

void vpux::VPURT::DeclareBufferOp::build(mlir::OpBuilder& builder, ::mlir::OperationState& state, mlir::Type type,
                                         VPURT::BufferSection section, ArrayRef<int64_t> sectionIndex,
                                         int64_t byteOffset) {
    build(builder, state, type, VPURT::BufferSectionAttr::get(builder.getContext(), section),
          getIntArrayAttr(builder, sectionIndex), getIntAttr(builder, byteOffset));
}

void vpux::VPURT::DeclareBufferOp::build(mlir::OpBuilder& builder, ::mlir::OperationState& state, mlir::Type type,
                                         VPURT::BufferSection section, int64_t sectionIndex, int64_t byteOffset) {
    build(builder, state, type, VPURT::BufferSectionAttr::get(builder.getContext(), section),
          getIntArrayAttr(builder, makeArrayRef({sectionIndex})), getIntAttr(builder, byteOffset));
}

mlir::LogicalResult vpux::VPURT::verifyOp(DeclareBufferOp op) {
    const auto type = op.getType().cast<vpux::NDTypeInterface>();
    const auto section = op.section();

    if (!VPURT::isMemoryCompatible(section, type)) {
        return errorAt(op, "BufferSection '{0}' is not compatible with memory space '{1}'", section,
                       type.getMemSpace());
    }

    const auto maybeSectionIndex = op.sectionIndex();
    if (maybeSectionIndex.hasValue()) {
        if (maybeSectionIndex.getValue().empty()) {
            return errorAt(op, "Empty section index is not supported");
        }
    }

    if (op.section() == VPURT::BufferSection::CMX_NN) {
        const auto checkSectionIndex = [&op, &type](ArrayRef<int64_t> sectionIdx) {
            if (auto distributedType = type.dyn_cast<VPUIP::DistributedBufferType>()) {
                const auto distribution = distributedType.getDistribution();
                const auto numClusters = checked_cast<size_t>(distribution.num_clusters().getInt());
                if (numClusters != sectionIdx.size()) {
                    return errorAt(op, "Number of clusters '{0}' and section indexes '{1}' mismatch", numClusters,
                                   sectionIdx.size());
                }
            }

            if (sectionIdx.size() == 1) {
                const auto memSpace = type.getMemSpace();
                if (memSpace == nullptr) {
                    return errorAt(op, "Output type must have CMX_NN memory space");
                }

                const auto maybeIdx = memSpace.getIndex();
                if (!maybeIdx.hasValue()) {
                    return errorAt(op, "Output type must have memory space index equal to '{0}', but it doesn't",
                                   sectionIdx[0]);
                }

                const auto memSpaceIdx = maybeIdx.getValue();
                if (memSpaceIdx != sectionIdx[0]) {
                    return errorAt(op, "Section index '{0}' and memory space index '{1}' mismatch", sectionIdx[0],
                                   memSpaceIdx);
                }
            }

            if (sectionIdx.size() > 1) {
                const auto distributedType = type.dyn_cast<VPUIP::DistributedBufferType>();
                if (distributedType == nullptr) {
                    return errorAt(op, "Array of section indexes is supported only for distributed buffer type");
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
    } else if (op.section() == VPURT::BufferSection::DDR) {
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

//
// DeclareSparseBufferOp
//

mlir::ValueRange vpux::VPURT::DeclareSparseBufferOp::getViewSources() {
    return getOperands();
}

void vpux::VPURT::DeclareSparseBufferOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                               mlir::Value data) {
    const auto inputType = data.getType().cast<mlir::MemRefType>();
    const auto sparseBuffer = VPURT::SparseBufferType::get(inputType);
    build(odsBuilder, odsState, sparseBuffer, data, nullptr, nullptr);
}

void vpux::VPURT::DeclareSparseBufferOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                               mlir::Value data, mlir::Value sparsityMap) {
    const auto inputType = data.getType().cast<mlir::MemRefType>();
    const auto sparsityMapType = sparsityMap.getType().cast<mlir::MemRefType>();
    const auto sparseBuffer = VPURT::SparseBufferType::get(inputType, sparsityMapType);
    build(odsBuilder, odsState, sparseBuffer, data, sparsityMap, nullptr);
}

void vpux::VPURT::DeclareSparseBufferOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                               mlir::Value data, mlir::Value sparsityMap,
                                               mlir::Value storageElementTable) {
    const auto inputType = data.getType().cast<mlir::MemRefType>();
    const auto sparsityMapType = sparsityMap.getType().cast<mlir::MemRefType>();
    const auto storageElementTableType = storageElementTable.getType().cast<mlir::MemRefType>();
    const auto sparseBuffer = VPURT::SparseBufferType::get(inputType, sparsityMapType, storageElementTableType);
    build(odsBuilder, odsState, sparseBuffer, data, sparsityMap, storageElementTable);
}
