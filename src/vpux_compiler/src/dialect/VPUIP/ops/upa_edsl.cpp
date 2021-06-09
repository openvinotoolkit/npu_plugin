//
// Copyright 2021 Intel Corporation.
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

#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/edsl/kernel_gen.hpp"
#include "vpux/compiler/edsl/utils.hpp"

#include <mlir/IR/AffineExprVisitor.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/DebugStringHelper.h>

#ifdef ENABLE_PLAIDML

#include "pmlc/util/strides.h"

using namespace vpux;
using namespace mlir;

VPUIP::BlobWriter::SpecificTask VPUIP::EdslUPAOp::serialize(VPUIP::BlobWriter& writer) {
    // Kernel binary
    SymbolRefAttr kernelRef = kernel();
    auto module = (*this)->getParentOfType<ModuleOp>();
    auto func = module.lookupSymbol<FuncOp>(kernelRef);
    VPUX_THROW_UNLESS(func != nullptr, "Could not resolve kernel symbol reference '{0}'", kernelRef);
    edsl::MoviCompileParams params = {
            /*cpu=*/"3010xx",
            /*moviCompile=*/"linux64/bin/moviCompile",
            /*mdkLinker=*/"linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-ld",
            /*mdkLibDir=*/"common/moviCompile/lib/30xxxx-leon",
            /*mdkLibs=*/
            {
                    "mlibcxx.a",
                    "mlibneon.a",
                    "mlibVecUtils.a",
                    "mlibm.a",
                    "mlibc_lite.a",
                    "mlibc_lite_lgpl.a",
                    "mlibcrt.a",
            },
    };
    flatbuffers::Offset<MVCNN::BinaryData> elfBinary = edsl::generateKernelForSHAVE(func, params, writer);

    // Outers and middles
    SmallVector<uint32_t, 4> outerRanges = edsl::getVectorFromArrayAttr<uint32_t>(outers());
    size_t numOuters = outerRanges.size();
    SmallVector<uint32_t, 4> outerSteps(numOuters, 1);
    SmallVector<uint32_t, 4> middleRanges = edsl::getVectorFromArrayAttr<uint32_t>(middles());
    size_t numMiddles = middleRanges.size();
    SmallVector<uint32_t, 4> middleSteps(numMiddles, 1);

    auto interiorOperands = func.getArguments().drop_front(numOuters + numMiddles);

    // Init values
    SmallVector<MVCNN::InitValue, 4> initOutputs;
    for (auto init : inits()) {
        initOutputs.emplace_back(edsl::convertInitValue(init));
    }

    // DMA descriptors
    SmallVector<uint8_t, 4> inputTypes;
    SmallVector<uint8_t, 4> outputTypes;
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
        SmallVector<uint32_t, 4> exteriorShape = edsl::padShapeTo4Dim(exteriorType.getShape());
        auto interiorType = interior.getType().cast<MemRefType>();
        SmallVector<uint32_t, 4> interiorShape = edsl::padShapeTo4Dim(interiorType.getShape());
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
        auto schemaType = edsl::getSchemaDataType(elementType);
        if (fromDDR) {
            inputTypes.emplace_back(schemaType);
        } else {
            outputTypes.emplace_back(schemaType);
        }
        int64_t bufArg = operand.index();
        int64_t stage = static_cast<int64_t>(dmaDesc.stage().getValue());
        int64_t dataTypeSize = Byte(getElemTypeSize(elementType)).count();
        auto dmaTransfer = MVCNN::CreateDmaTransfer(writer, fromDDR, dataTypeSize, bufArg, stage, localShape,
                                                    globalShape, ranges, basePoly);
        dmaTransfers.emplace_back(dmaTransfer);
    }

    const auto outerRangesOff = writer.createVector(outerRanges);
    const auto outerStepsOff = writer.createVector(outerSteps);
    const auto middleRangesOff = writer.createVector(middleRanges);
    const auto middleStepsOff = writer.createVector(middleSteps);
    const auto inputTypesOff = writer.createVector(inputTypes);
    const auto outputTypesOff = writer.createVector(outputTypes);
    const auto initOutputsOff = writer.createVectorOfStructs<MVCNN::InitValue>(initOutputs);
    const auto dmaTransfersOff = writer.createVector(dmaTransfers);

    MVCNN::EdslParamsBuilder builder(writer);
    builder.add_outerRanges(outerRangesOff);
    builder.add_outerSteps(outerStepsOff);
    builder.add_middleRanges(middleRangesOff);
    builder.add_middleSteps(middleStepsOff);
    builder.add_inputDataTypes(inputTypesOff);
    builder.add_outputDataTypes(outputTypesOff);
    builder.add_initOutputs(initOutputsOff);
    builder.add_dmaTransfers(dmaTransfersOff);
    builder.add_kernelData(elfBinary);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_EdslParams});
}

#else  // ENABLE_PLAIDML

vpux::VPUIP::BlobWriter::SpecificTask vpux::VPUIP::EdslUPAOp::serialize(VPUIP::BlobWriter& /*writer*/) {
    VPUX_THROW("EdslUPAOp is only supported when ENABLE_PLAIDML=ON");
}

#endif  // ENABLE_PLAIDML

void vpux::VPUIP::EdslUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange inputs,
                                   mlir::ValueRange outputs, ::mlir::SymbolRefAttr kernel, ::mlir::ArrayAttr outers,
                                   ::mlir::ArrayAttr middles, ::mlir::ArrayAttr inits, ::mlir::ArrayAttr transfers) {
    build(builder, state,
          /*resultTypes=*/outputs.getTypes(),
          /*inputs=*/inputs,
          /*outputs=*/outputs,
          /*waitBarriers=*/mlir::ValueRange{},
          /*updateBarriers=*/mlir::ValueRange{},
          /*kernel=*/kernel,
          /*outers=*/outers,
          /*middles=*/middles,
          /*inits=*/inits,
          /*transfers=*/transfers,
          /*maxShaves=*/nullptr,
          /*isTrailingSWLayer=*/false);
}
