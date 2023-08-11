//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <cstddef>

namespace {

template <typename OpType>
size_t getEntryBlockSize(mlir::Operation* operation) {
    if (!operation->getNumRegions())
        return 0;

    auto& blocks = operation->getRegion(0).getBlocks();
    if (blocks.empty())
        return 0;

    auto ops = blocks.front().getOps<OpType>();
    return vpux::checked_cast<size_t>(std::distance(ops.begin(), ops.end()));
}

template <typename ChildOpType, typename ParentOpType>
bool isValidChild(ParentOpType& parentOp, vpux::FuncRef<bool(size_t)> predicate) {
    static_assert(ChildOpType::template hasTrait<::mlir::OpTrait::template HasParent<ParentOpType>::template Impl>(),
                  "ChildOpType does not belong to ParentOpType");

    return predicate(getEntryBlockSize<ChildOpType>(parentOp.getOperation()));
}

// base recursion function - needed as recursion terminator
template <typename ParentOpType>
bool areValidChildren(ParentOpType&, vpux::FuncRef<bool(size_t)>) {
    return true;
}

template <typename ParentOpType, typename FirstChildOpType, typename... RemainingChildOpTypes>
bool areValidChildren(ParentOpType& p, vpux::FuncRef<bool(size_t)> predicate) {
    if (!isValidChild<FirstChildOpType>(p, predicate)) {
        return false;
    }

    return areValidChildren<ParentOpType, RemainingChildOpTypes...>(p, predicate);
}

}  // namespace

namespace vpux {

template <typename ParentOpType, typename... ChildOpTypes>
bool hasMandatorySingleInstanceChildren(ParentOpType& parentOp) {
    return areValidChildren<ParentOpType, ChildOpTypes...>(parentOp, [](size_t numChildOpInstances) {
        return numChildOpInstances == 1;
    });
}

template <typename ParentOpType, typename... ChildOpTypes>
bool hasOptionalSingleInstanceChildren(ParentOpType& parentOp) {
    return areValidChildren<ParentOpType, ChildOpTypes...>(parentOp, [](size_t numChildOpInstances) {
        return numChildOpInstances <= 1;
    });
}

}  // namespace vpux
