#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"

using namespace vpux;

mlir::LogicalResult VPURegMapped::GroupYieldOp::verify() {
    if (getListHeads().size() != getListTails().size()) {
        return emitOpError() << " expects listHeads " << getListHeads() << " and listTails " << getListTails()
                             << " to be of same size";
    }

    auto groupOp = getOperation()->getParentOfType<VPURegMapped::ExecutionGroupOp>();
    if (!groupOp) {
        return emitOpError() << " expects parent op " << VPURegMapped::ExecutionGroupOp::getOperationName();
    }

    if (groupOp.getEndIndexes().size() != getListTails().size()) {
        return emitOpError() << " expects listTails " << getListTails() << " and parent endIndexes "
                             << groupOp.getEndIndexes() << " to be of same size ";
    }

    if (groupOp.getStartIndexes().size() != getListHeads().size()) {
        return emitOpError() << " expects listHeads " << getListHeads() << " and parent startIndexes"
                             << groupOp.getStartIndexes() << " to be of same size ";
    }

    for (auto [endIdx, listTail, startIdx, listHead] :
         llvm::zip_equal(groupOp.getEndIndexes(), getListTails(), groupOp.getStartIndexes(), getListHeads())) {
        if (endIdx.getType() != listTail.getType()) {
            return emitOpError() << " expects listTails " << getListTails() << " and GroupOp endIndexes"
                                 << groupOp.getEndIndexes() << " to be of the same type";
        }

        if (startIdx.getType() != listHead.getType()) {
            return emitOpError() << " expect listHeads " << getListHeads() << " and groupOp startIndexes"
                                 << groupOp.getStartIndexes() << " to be of the same type";
        }
    }

    return mlir::success();
}

mlir::LogicalResult VPURegMapped::ExecutionGroupOp::verify() {
    if (getStartIndexes().size() != getEndIndexes().size()) {
        return emitOpError() << " expects startIndexes " << getStartIndexes() << " and endIdexes " << getEndIndexes()
                             << " to be of same size";
    }

    auto block = &getTasks().front();

    if (block->getArguments().size() != getPreviousTaskIdx().size()) {
        return emitOpError() << "expects previousOpIndexes " << getPreviousTaskIdx() << " and blockArguments "
                             << block->getArguments() << " to be of the same size";
    }

    for (auto [prevIdx, arg] : llvm::zip_equal(getPreviousTaskIdx(), block->getArguments())) {
        if (prevIdx.getType() != arg.getType()) {
            return emitOpError() << " expects startIndexes " << getStartIndexes() << " and blockArguments "
                                 << block->getArguments() << "to be of the same types";
        }
    }

    return mlir::success();
}
