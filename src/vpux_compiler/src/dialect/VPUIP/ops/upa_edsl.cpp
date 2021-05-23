//
// Copyright 2021 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/DebugStringHelper.h"

#ifdef ENABLE_PLAIDML
#include "pmlc/util/strides.h"

using namespace vpux;
using namespace mlir;

static unsigned getByteWidth(Type type) {
    unsigned width = type.getIntOrFloatBitWidth();
    return llvm::divideCeil(width, 8);
}

template <typename Type>
static SmallVector<Type, 4> getVectorFromArrayAttr(ArrayAttr attrs) {
    SmallVector<Type, 4> result;
    for (Attribute attr : attrs.getValue()) {
        result.emplace_back(attr.cast<IntegerAttr>().getInt());
    }
    return result;
}

static SmallVector<uint32_t, 4> convertShape(ArrayRef<int64_t> from) {
    SmallVector<uint32_t, 4> into;
    if (from.size() <= 4) {
        for (unsigned i = 0; i < 4 - from.size(); ++i) {
            into.emplace_back(1);
        }
    }
    std::copy(from.begin(), from.end(), std::back_inserter(into));
    return into;
}

VPUIP::BlobWriter::SpecificTask VPUIP::EdslUPAOp::serialize(VPUIP::BlobWriter& writer) {
    // Outers and middles
    SmallVector<uint32_t, 4> outerRanges = getVectorFromArrayAttr<uint32_t>(outers());
    size_t numOuters = outerRanges.size();
    SmallVector<uint32_t, 4> outerSteps(numOuters, 1);
    SmallVector<uint32_t, 4> middleRanges = getVectorFromArrayAttr<uint32_t>(middles());
    size_t numMiddles = middleRanges.size();
    SmallVector<uint32_t, 4> middleSteps(numMiddles, 1);

    // Kernel binary
    SymbolRefAttr kernelRef = kernel();
    auto module = (*this)->getParentOfType<ModuleOp>();
    auto func = module.lookupSymbol<FuncOp>(kernelRef);
    if (!func) {
        throw std::runtime_error("Could not resolve kernel symbol reference: " + debugString(kernelRef));
    }
    auto interiorOperands = func.getArguments().drop_front(numOuters + numMiddles);

    // DMA descriptors
    SmallVector<flatbuffers::Offset<MVCNN::DmaTransfer>> dmaTransfers;
    SmallVector<Value> exteriorOperands;
    for (Value value : inputs()) {
        exteriorOperands.push_back(value);
    }
    for (Value value : outputs()) {
        exteriorOperands.push_back(value);
    }
    auto zipped = llvm::zip(exteriorOperands, interiorOperands, transfers());
    for (auto operand : llvm::enumerate(zipped)) {
        Value exterior, interior;
        Attribute attr;
        std::tie(exterior, interior, attr) = operand.value();

        auto exteriorType = exterior.getType().cast<MemRefType>();
        SmallVector<uint32_t, 4> exteriorShape = convertShape(exteriorType.getShape());
        auto interiorType = interior.getType().cast<MemRefType>();
        SmallVector<uint32_t, 4> interiorShape = convertShape(interiorType.getShape());
        Type elementType = interiorType.getElementType();

        auto dmaDesc = attr.cast<VPUIP_EdslDMADesc>();
        AffineMap baseMap = dmaDesc.baseMap().getValue();
        auto strideArray = pmlc::util::computeStrideArray(baseMap);
        std::vector<MVCNN::Term> terms;
        for (auto stride : llvm::enumerate(strideArray->strides)) {
            terms.emplace_back(stride.value(), stride.index());
        }
        int64_t constant = strideArray->offset;
        auto basePoly = MVCNN::CreatePolynomialDirect(writer, &terms, constant);
        auto localShape = writer.createVector(interiorShape);
        auto globalShape = writer.createVector(exteriorShape);
        auto ranges = writer.createVector(interiorShape);
        bool fromDDR =
                dmaDesc.dir().getValue() == EdslDMADirection::IN || dmaDesc.dir().getValue() == EdslDMADirection::INOUT;
        int64_t bufArg = operand.index();
        int64_t stage = static_cast<int64_t>(dmaDesc.stage().getValue());
        int64_t dataTypeSize = getByteWidth(elementType);
        auto dmaTransfer = MVCNN::CreateDmaTransfer(writer, fromDDR, dataTypeSize, bufArg, stage, localShape,
                                                    globalShape, ranges, basePoly);
        dmaTransfers.emplace_back(dmaTransfer);
    }

    MVCNN::EdslParamsBuilder builder(writer);
    builder.add_outerRanges(writer.createVector(outerRanges));
    builder.add_outerSteps(writer.createVector(outerSteps));
    builder.add_middleRanges(writer.createVector(middleRanges));
    builder.add_middleSteps(writer.createVector(middleSteps));
    builder.add_dmaTransfers(writer.createVector(dmaTransfers));
    // TODO: compile kernel() and get the binary data
    // builder.add_kernelData(elfBinary);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_EdslParams});
}

#else  // ENABLE_PLAIDML

vpux::VPUIP::BlobWriter::SpecificTask vpux::VPUIP::EdslUPAOp::serialize(VPUIP::BlobWriter& /*writer*/) {
    throw std::runtime_error("EdslUPAOp is only supported when ENABLE_PLAIDML=ON");
}

#endif  // ENABLE_PLAIDML

void vpux::VPUIP::EdslUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange inputs,
                                   mlir::ValueRange outputs, ::mlir::SymbolRefAttr kernel, ::mlir::ArrayAttr outers,
                                   ::mlir::ArrayAttr middles, ::mlir::ArrayAttr transfers) {
    build(builder, state,
          /*resultTypes=*/outputs.getTypes(),
          /*inputs=*/inputs,
          /*outputs=*/outputs,
          /*waitBarriers=*/mlir::ValueRange{},
          /*updateBarriers=*/mlir::ValueRange{},
          /*kernel=*/kernel,
          /*outers=*/outers,
          /*middles=*/middles,
          /*transfers=*/transfers,
          /*maxShaves=*/nullptr,
          /*isTrailingSWLayer=*/false);
}
